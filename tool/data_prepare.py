#用于将FS2K数据集下的素描图片按照标注文件划分到data文件夹下的train、test子文件夹中
import csv
import os
import json
import shutil

if not os.path.exists('../data'):
    os.mkdir('../data')
    if not os.path.exists('../data/train'):
        os.mkdir('../data/train')
    if not os.path.exists('../data/test'):
        os.mkdir('../data/test')

attribute_list = ["hair","gender","earring","smile","frontal_face","style"]#指定用到的属性列表
root_dir = '../FS2K'
root_dir_photo = os.path.join(root_dir, 'photo')
root_dir_sketch = os.path.join(root_dir, 'sketch')

json_file_train = os.path.join(root_dir, 'anno_train.json')
json_file_test = os.path.join(root_dir, 'anno_test.json')

with open(json_file_train,'r') as f1:
    train_json_data = json.loads(f1.read())
with open(json_file_test,'r') as f2:
    test_json_data = json.loads(f2.read())



with open('../data/train_anno.csv','w',encoding='utf-8',newline='\n') as fw1:
    csv_writer = csv.DictWriter(fw1,fieldnames=["name","hair","gender","earring","smile","frontal_face","style"])
    csv_writer.writeheader()
    train_count = 1
    for idx_fs, fs in enumerate(train_json_data):
        dic = {}
        img = fs["image_name"]
        sub_dir,img_name = img.split('/')
        sub_dir,img_name = sub_dir.replace('photo', 'sketch'),img_name.replace('image', 'sketch')
        if sub_dir =='sketch2':#因为只有sketch2文件夹下图片格式为png
            img_name += '.png'
            target_name = '{:0>4d}'.format(train_count) +'.png'
        else:
            img_name += '.jpg'
            target_name = '{:0>4d}'.format(train_count) + '.jpg'
        shutil.copyfile(os.path.join(root_dir,'sketch',sub_dir,img_name),os.path.join('../data/train',target_name))
        dic["name"] = target_name
        for attr in attribute_list:
            dic[attr] = fs[attr]
        #print(idx_fs)
        csv_writer.writerow(dic)
        train_count += 1
with open('../data/test_anno.csv','w',encoding='utf-8',newline='') as fw1:
    csv_writer = csv.DictWriter(fw1,fieldnames=["name","hair","gender","earring","smile","frontal_face","style"])
    csv_writer.writeheader()
    test_count = 1
    for idx_fs, fs in enumerate(train_json_data):
        dic = {}
        img = fs["image_name"]
        sub_dir, img_name = img.split('/')
        sub_dir, img_name = sub_dir.replace('photo', 'sketch'), img_name.replace('image', 'sketch')
        if sub_dir == 'sketch2':  # 因为只有sketch2文件夹下图片格式为png
            img_name += '.png'
            target_name = '{:0>4d}'.format(test_count) + '.png'
        else:
            img_name += '.jpg'
            target_name = '{:0>4d}'.format(test_count) + '.jpg'
        shutil.copyfile(os.path.join(root_dir, 'sketch', sub_dir, img_name), os.path.join('../data/test', target_name))
        dic["name"] = target_name
        for attr in attribute_list:
            dic[attr] = fs[attr]
        csv_writer.writerow(dic)
        test_count += 1