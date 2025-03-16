import math
from sqlite3 import paramstyle
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from abc import abstractmethod
from .bdc_module import *
from einops import rearrange
# from EVLdecoder import TemporalCrossAttention, EVLDecoder
from .tools.svPN import svPN
from .tools.MPN import MPN
from .tools.ePN import ePN
from .tools.scaleNorm import scaleNorm
from .tools import MPNCOV

class Covariance(nn.Module):
    def __init__(self,
                remove_mean=True,
                conv=False,
        ):
        super(Covariance, self).__init__()
        self.remove_mean = remove_mean
        self.conv = conv

    def _remove_mean(self, x):
        x = x.transpose(-1, -2)
        _mean = F.adaptive_avg_pool2d(x, (1,1))
        x = x - _mean
        x = x.transpose(-1, -2)
        return x

    def remove_mean_(self, x):
        _mean = F.adaptive_avg_pool2d(x, (1,1))
        x = x - _mean
        return x

    def _cov(self, x):
        batchsize, d, N = x.size()
        x = x.transpose(-1, -2)
        y = (1. / N ) * (x.bmm(x.transpose(1, 2)))
        return y
    
    def _cross_cov(self, x1, x2):
        batchsize1, N1, d1 = x1.size()
        batchsize2, N2, d2 = x2.size()
        assert batchsize1 == batchsize2
        assert N1 == N2
        x1 = x1.transpose(-1, -2)
        x2 = x2.transpose(-1, -2)

        y = (1. / N1) * (x1.bmm(x2.transpose(1, 2)))
        return y
    
    def cross_cov(self, x1, x2):
        batchsize1, d1, h1, w1 = x1.size()
        batchsize2, d2, h2, w2 = x2.size()
        N1 = h1*w1
        N2 = h2*w2
        assert batchsize1 == batchsize2
        assert N1 == N2
        x1 = x1.view(batchsize1, d1, N1)
        x2 = x2.view(batchsize2, d2, N2)

        y = (1. / N1) * (x1.bmm(x2.transpose(1, 2)))
        return y

    def forward(self, x, y=None):
        if self.remove_mean:
            x = self.remove_mean_(x) if self.conv else self._remove_mean(x)
            if y is not None:
                y = self.remove_mean_(y) if self.conv else self._remove_mean(y)          
        if y is not None:
            if self.conv:
                x = self.cross_cov(x, y)
            else:
                x = self._cross_cov(x, y)
        else:
            x = self._cov(x)
        return x

class BaselineTrain(nn.Module):
    def __init__(self, params, model_func, num_class):
        super(BaselineTrain, self).__init__()
        self.params = params
        self.feature = model_func()
        # print('feature',self.feature)  #resnet
        if params.method in ['stl_deepbdc', 'meta_deepbdc']:
            reduce_dim = params.reduce_dim
            self.feat_dim = int(reduce_dim * (reduce_dim+1) / 2)
            self.dcov = BDC(is_vec=True, input_dim=self.feature.feat_dim, dimension_reduction=reduce_dim)
            self.dropout = nn.Dropout(params.dropout_rate)

        elif params.method in ['protonet', 'good_embed']:
            self.feat_dim = self.feature.feat_dim[0]
            self.avgpool = nn.AdaptiveAvgPool2d(1)

        if params.method in ['stl_deepbdc', 'meta_deepbdc', 'protonet', 'good_embed']:
            self.classifier = nn.Linear(self.feat_dim, num_class)
            self.classifier.bias.data.fill_(0)

        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()

    def feature_forward(self, x):
        # print('x',x.shape)  #x torch.Size([64, 3, 224, 224])
        out = self.feature.forward(x)
        # print('out',out.shape) #out torch.Size([64, 512, 14, 14])
        if self.params.method in ['stl_deepbdc', 'meta_deepbdc']:
            out = self.dcov(out)
            # print('out',out.shape)  #out torch.Size([64, 8256])
            out = self.dropout(out)
            # print('out',out.shape) #out torch.Size([64, 8256])
        elif self.params.method in ['protonet', 'good_embed']:
            out = self.avgpool(out).view(out.size(0), -1)
        return out

    def forward(self, x):
        
        x = Variable(x.cuda())
        out = self.feature_forward(x)
        # print('out2',out.shape)  #out torch.Size([64, 8256])
        scores = self.classifier.forward(out)
        # print('scores',scores.shape)  #  scores torch.Size([64, 200])
        return scores

    def forward_meta_val(self, x):
        x = Variable(x.cuda())        
        x = x.contiguous().view(self.params.val_n_way * (self.params.n_shot + self.params.n_query), *x.size()[2:])
        
        out = self.feature_forward(x)

        z_all = out.view(self.params.val_n_way, self.params.n_shot + self.params.n_query, -1)
        z_support = z_all[:, :self.params.n_shot]
        z_query = z_all[:, self.params.n_shot:]
        z_proto = z_support.contiguous().view(self.params.val_n_way, self.params.n_shot, -1).mean(1)
        z_query = z_query.contiguous().view(self.params.val_n_way * self.params.n_query, -1)

        if self.params.method in ['meta_deepbdc']:
            scores = self.metric(z_query, z_proto)
        elif self.params.method in ['protonet']:
            scores = self.euclidean_dist(z_query, z_proto)
        return scores

    def forward_loss(self, x, y):
        scores = self.forward(x)
        y = Variable(y.cuda())
        return self.loss_fn(scores, y), scores

    def forward_meta_val_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.params.val_n_way), self.params.n_query))
        y_query = Variable(y_query.cuda())
        y_label = np.repeat(range(self.params.val_n_way), self.params.n_query)
        scores = self.forward_meta_val(x)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_label)
        return float(top1_correct), len(y_label), self.loss_fn(scores, y_query), scores

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 2000
        avg_loss = 0
        total_correct = 0

        iter_num = len(train_loader)
        print('iter_num',iter_num)
        total = len(train_loader) * self.params.batch_size

        for i, (x, y) in enumerate(train_loader):
            y = Variable(y.cuda())
            optimizer.zero_grad()
            # print('x3',x.shape)  #x3 torch.Size([64, 3, 224, 224])
            loss, output = self.forward_loss(x, y)
            # print('xoutput3',output.shape) #xoutput3 torch.Size([64, 200])
            pred = output.data.max(1)[1]
            total_correct += pred.eq(y.data.view_as(pred)).sum()
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.item()

            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss / float(i + 1)))
        return avg_loss / iter_num, float(total_correct) / total * 100

    def test_loop(self, val_loader):
        total_correct = 0
        avg_loss = 0.0
        total = len(val_loader) * self.params.batch_size
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                y = Variable(y.cuda())
                loss, output = self.forward_loss(x, y)
                avg_loss = avg_loss + loss.item()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(y.data.view_as(pred)).sum()
        avg_loss /= len(val_loader)
        acc = float(total_correct) / total
        # print('Test Acc = %4.2f%%, loss is %.2f' % (acc * 100, avg_loss))
        return avg_loss, acc * 100

    def meta_test_loop(self, test_loader):
        acc_all = []
        avg_loss = 0
        iter_num = len(test_loader)
        with torch.no_grad():
            for i, (x, _) in enumerate(test_loader):
                correct_this, count_this, loss, _ = self.forward_meta_val_loss(x)
                acc_all.append(correct_this / count_this * 100)
                avg_loss = avg_loss + loss.item()
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        return avg_loss / iter_num, acc_mean

    def metric(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        if self.params.n_shot > 1:
            dist = torch.pow(x - y, 2).sum(2)
            score = -dist
        else:
            score = (x * y).sum(2)
        return score
    
    def euclidean_dist(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        score = -torch.pow(x - y, 2).sum(2)
        return score
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class MetaTemplate(nn.Module):
    def __init__(self, params, model_func, n_way, n_support, change_way=True):
        super(MetaTemplate, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = params.n_query  # (change depends on input)
        self.feature = model_func()
        self.change_way = change_way  # some methods allow different_way classification during training and test
        self.params = params
        class_dim = 384  
        self.class_dim = 384 

        #number of group G
        self.num_heads = 4
        num_heads = self.num_heads
        # self.head2 = nn.Linear(2080, class_dim)
        self.head2 = nn.Linear(4096, class_dim)

        # self.class_head='multihead_cov'
        self.class_head='temporal_cov' #!!!

        # self.class_head='non'
        self.class_way='add'
        # self.class_way='score'

        self.layer_reduce1 = nn.Conv2d(
            class_dim,
            int(class_dim/3),
            kernel_size=1,
            stride=[1,1],
            padding=0,
            bias=False,
         )

        self.layer_reduce_bn1 = nn.BatchNorm2d(
            num_features=(int(class_dim/3)),
         )

        self.layer_reduce2 = nn.Conv2d(
            int(class_dim/3),
            int(class_dim/6),
            kernel_size=1,
            stride=[1,1],
            padding=0,
            bias=False,
         )

        self.layer_reduce_bn2 = nn.BatchNorm2d(
            num_features=int(class_dim/6),
         )
        self.layer_reduce3 = nn.Conv2d(
            num_heads,
            int(num_heads/2),
            kernel_size=3,
            stride=[1,1],
            padding=1,
            bias=False,
         )
        self.layer_reduce_bn3 = nn.BatchNorm2d(
            num_features=int(num_heads/2),
         )
        self.layer_reduce4 = nn.Conv2d(
            int(num_heads/2),
            int(num_heads/4),
            kernel_size=3,
            stride=[2,2],
            padding=1,
            bias=False,
         )
        self.layer_reduce_bn4 = nn.BatchNorm2d(
            num_features=int(num_heads/4),
         )
        self.relu_op = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
    @abstractmethod
    def set_forward(self, x, is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    @abstractmethod
    def feature_forward(self, x):
        pass

    def forward(self, x):
        out = self.feature.forward(x)
        return out

    def parse_feature(self, x, is_feature):
        x = Variable(x.cuda())
        _, _, _, _, _, image_size = x.shape
        if is_feature:
            z_all = x
        else:
            x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:]) #torch.Size([85, 3, 16, 224, 224])

            if self.class_head=='temporal_cov' and self.class_way=='add':
                x, y = self.feature.forward(x)#x,torch.Size([85, 768])
                B,N,C=y.shape
                T = 8
                y = y.permute(0,2,1)
                if image_size == 224:
                    y = y.reshape(B,C,int(N/14),14) #224*224
                elif image_size == 112:
                    y = y.reshape(B,C,int(N/7),7) #112
                else:
                    print("image_size is wrong")
                    assert 0==1

                y = self.layer_reduce1(y)
                y = self.layer_reduce_bn1(y)
                y = self.relu_op(y)

                y = self.layer_reduce2(y)
                y = self.layer_reduce_bn2(y)
                y = self.relu_op(y)
                
                # ELSTC module
                B,C,N,H=y.shape
                y = y.reshape(B,C*T,int(N/T),H).view(B, C, int(T/self.num_heads), self.num_heads, int(N/T), H)
                y = rearrange(y,'b c t n h w -> (b n) (c t) h w')

                y = MPNCOV.CovpoolLayer(y)
                y = MPNCOV.SqrtmLayer(y, 3)

                B,C, _ =  y.shape
                y = y.reshape(int(B/self.num_heads),self.num_heads,C,C)
                y = self.layer_reduce3(y)
                y = self.layer_reduce_bn3(y)
                y = self.relu_op(y)
                # print('y',y.shape)  #y torch.Size([85, 2, 128, 128])

                y = self.layer_reduce4(y)
                y = self.layer_reduce_bn4(y)
                y = self.relu_op(y)

                y = y.flatten(1)

                x2 = self.head2(y)

                z_all = x.view(self.n_way, self.n_support + self.n_query, -1) + x2.view(self.n_way, self.n_support + self.n_query, -1)

            if self.class_head=='temporal_cov' and self.class_way=='score':
                
                B, C, T, H, W = x.size()
                x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
                # print('x',x.shape)
                # assert 0==1
                x = self.feature.forward(x) #x,torch.Size([85, 768])
                
                # print('x',x.shape)
                x = x.reshape(B,8*197,C)
                y=x
                y = y.reshape(B,8,197,C)[:,:,1:,:]
                y = y.reshape(B,8*196,C)
                x = x.permute(0, 2, 1)
                x = self.avgpool(x).permute(0, 2, 1).reshape(B,C)
                
                B,N,C=y.shape
                y = y.permute(0,2,1)
                y = y.reshape(B,C,int(N/14),14)
                # print('y1',y.shape)

                y = self.layer_reduce1(y)
                y = self.layer_reduce_bn1(y)
                y = self.relu_op(y)

                y = self.layer_reduce2(y)
                y = self.layer_reduce_bn2(y)
                y = self.relu_op(y)

                y = MPNCOV.CovpoolLayer(y)
                # fast matrix power normalization, P_{TCP}^{1/2}
                y = MPNCOV.SqrtmLayer(y, 3)
                y = MPNCOV.TriuvecLayer(y)
                y = y.unsqueeze(-1)

                y = y.view(x.size(0), -1)
                
                x2 = y
                z_all2 = x2.view(self.n_way, self.n_support + self.n_query, -1)

                z_all1 = x.view(self.n_way, self.n_support + self.n_query, -1) 

            if self.class_head=='non':
                z_all,y = self.feature.forward(x)#x,torch.Size([85, 768])
                z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)

        if self.class_head=='non':
            z_support = z_all[:, :self.n_support]
            z_query = z_all[:, self.n_support:]

            return z_support, z_query  
        if self.class_way=='add':
            z_support = z_all[:, :self.n_support]
            z_query = z_all[:, self.n_support:]

            return z_support, z_query, 1, 1
        if self.class_way=='score':
            # z_all = z_all1 + z_all2
            z_support1 = z_all1[:, :self.n_support]
            z_query1 = z_all1[:, self.n_support:]
            z_support2 = z_all2[:, :self.n_support]
            z_query2 = z_all2[:, self.n_support:]
            return z_support1, z_query1,z_support2, z_query2

    def correct(self, x):
        scores = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)

    def train_loop(self, epoch, train_loader, optimizer):

        print_freq = 30
        avg_loss = 0
        acc_all = []
        iter_num = len(train_loader)
        for i, (x, _) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            optimizer.zero_grad()
            correct_this, count_this, loss, _ = self.set_forward_loss(x)
            acc_all.append(correct_this / count_this * 100)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()

            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        return avg_loss / iter_num, acc_mean

    def test_loop(self, test_loader, record=None):
        acc_all = []
        avg_loss = 0
        iter_num = len(test_loader)
        with torch.no_grad():
            for i, (x, _) in enumerate(test_loader):
                self.n_query = x.size(1) - self.n_support
                if self.change_way:
                    self.n_way = x.size(0)
                correct_this, count_this, loss, _ = self.set_forward_loss(x)
                acc_all.append(correct_this / count_this * 100)
                avg_loss = avg_loss + loss.item()
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        return avg_loss / iter_num, acc_mean


    