import json
import random

data = []
for i in range(0,64):
    data.append(i)
random.shuffle(data)
train_data = data[:48]
test_data = data[48:56]
val_data = data[56:64]


f = open("rare_cut.txt")
# f = open("/home/wyll/DeepBDC/filelist/rareact_cut/Rareact_train.txt")
names_list = []
labels_list = []
total =0
for line in f:
    # print(line)
    # assert 0==1
    iamge_labels, iamge_names = line.split('//')
    iamge_labels = int(iamge_labels[5:])
    # if iamge_labels % 8 == 0 or iamge_labels % 8 == 2 or iamge_labels % 8 == 7 or iamge_labels % 8 == 4 or iamge_labels % 8 == 5 or iamge_labels % 8 == 6:
    # if iamge_labels % 4 == 0 or iamge_labels % 4 == 2:
    if iamge_labels in train_data:

        iamge_names = iamge_names[:-1].strip()
        iamge_names = iamge_names[0:]
        iamge_names = '/' + iamge_names
        names_list.append(iamge_names)
        labels_list.append(iamge_labels)
        total = total +1
print(total)
dic = {'image_names': names_list, 'image_labels': labels_list}
with open('base.json', 'w') as F:
    json.dump(dic, F)

ff = open("rare_cut.txt")
names_list = []
labels_list = []
for line in ff:
    # print(line)
    # assert 0==1
    iamge_labels, iamge_names = line.split('//')
    iamge_labels = int(iamge_labels[5:])
    # if iamge_labels % 8 == 1:
    if iamge_labels in val_data:

        iamge_names = iamge_names[:-1].strip()
        iamge_names = iamge_names[0:]
        iamge_names = '/' + iamge_names
        names_list.append(iamge_names)
        labels_list.append(iamge_labels)

dic = {'image_names': names_list, 'image_labels': labels_list}
with open('val.json', 'w') as F:
    json.dump(dic, F)

fff = open("rare_cut.txt")
names_list = []
labels_list = []
for line in fff:
    # print(line)
    # assert 0==1
    iamge_labels, iamge_names = line.split('//')
    iamge_labels = int(iamge_labels[5:])
    # if iamge_labels % 8 == 3:
    if iamge_labels in test_data:

        iamge_names = iamge_names[:-1].strip()
        iamge_names = iamge_names[0:]
        iamge_names = '/' + iamge_names
        names_list.append(iamge_names)
        labels_list.append(iamge_labels)

dic = {'image_names': names_list, 'image_labels': labels_list}
with open('novel.json', 'w') as F:
    json.dump(dic, F)