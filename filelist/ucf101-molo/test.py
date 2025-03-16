import json
f = open("test_few_shot.txt")
names_list = []
labels_list = []
for line in f:
    # print(line)
    # assert 0==1
    iamge_labels, iamge_names = line.split('//')
    iamge_labels = int(iamge_labels[4:])
    iamge_names = iamge_names[:-1].strip()
    iamge_names = iamge_names[7:]
    iamge_names = '/home/wyll/DeepBDC/filelist/ucf101/UCF101/UCF-101/' + iamge_names
    print(iamge_names)
    names_list.append(iamge_names)
    labels_list.append(iamge_labels)

dic = {'image_names': names_list, 'image_labels': labels_list}
with open('novel.json', 'w') as F:
    json.dump(dic, F)
