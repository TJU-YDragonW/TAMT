import json
f = open("test_few_shot3.txt")
names_list = []
labels_list = []
for line in f:
    print(line)
    # assert 0==1
    iamge_labels, iamge_names = line.split('/')
    iamge_labels = int(iamge_labels[4:])
    iamge_names = iamge_names[:-1].strip()
    iamge_names = '/hd2/20bn-something-something-v2/' + iamge_names + '.webm'
    names_list.append(iamge_names)
    labels_list.append(iamge_labels)

dic = {'image_names': names_list, 'image_labels': labels_list}
with open('novel2.json', 'w') as F:
    json.dump(dic, F)
