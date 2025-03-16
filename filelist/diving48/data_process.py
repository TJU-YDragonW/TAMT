import json
f = open("D48_train.txt")

names_list = []
labels_list = []
for line in f:
    # print(line)
    # assert 0==1
    iamge_labels, iamge_names = line.split('//')
    iamge_labels = int(iamge_labels[5:])
    if iamge_labels % 8 == 0 or iamge_labels % 8 == 2 or iamge_labels % 8 == 3 or iamge_labels % 8 == 4 or iamge_labels % 8 == 5 or iamge_labels % 8 == 6:

        iamge_names = iamge_names[:-1].strip()
        iamge_names = iamge_names[0:]
        iamge_names = '/' + iamge_names
        names_list.append(iamge_names)
        labels_list.append(iamge_labels)

dic = {'image_names': names_list, 'image_labels': labels_list}
with open('base.json', 'w') as F:
    json.dump(dic, F)

ff = open("D48_train.txt")
names_list = []
labels_list = []
for line in ff:
    # print(line)
    # assert 0==1
    iamge_labels, iamge_names = line.split('//')
    iamge_labels = int(iamge_labels[5:])
    if iamge_labels % 8 == 1:

        iamge_names = iamge_names[:-1].strip()
        iamge_names = iamge_names[0:]
        iamge_names = '/' + iamge_names
        names_list.append(iamge_names)
        labels_list.append(iamge_labels)

dic = {'image_names': names_list, 'image_labels': labels_list}
with open('val.json', 'w') as F:
    json.dump(dic, F)

fff = open("D48_train.txt")
names_list = []
labels_list = []
for line in fff:
    # print(line)
    # assert 0==1
    iamge_labels, iamge_names = line.split('//')
    iamge_labels = int(iamge_labels[5:])
    if iamge_labels % 8 == 7:

        iamge_names = iamge_names[:-1].strip()
        iamge_names = iamge_names[0:]
        iamge_names = '/' + iamge_names
        names_list.append(iamge_names)
        labels_list.append(iamge_labels)

dic = {'image_names': names_list, 'image_labels': labels_list}
with open('novel.json', 'w') as F:
    json.dump(dic, F)