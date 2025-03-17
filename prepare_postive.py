import shutil
import random
import os
from tqdm import tqdm
origin_path = '../data/skull/skull_google'
target_path = '../data/skull/Skull_Infringement/postive'
origin_class_list = os.listdir(origin_path)

# bing_image_list = os.listdir(bing_path)
positive_image_list= []
for class_name in origin_class_list:
    class_path = os.path.join(origin_path, class_name)
    image_list = os.listdir(class_path)
    for i in range(len(image_list)):
        image_list[i] = os.path.join(class_name, image_list[i])
        # print(image_list[i])
    positive_image_list.extend(image_list)
    print('image list length',len(positive_image_list))
    # print(image_list[0])
    # input()

id = len(os.listdir(target_path))
print(id)
for image in positive_image_list:
    # print(os.path.join(trade_image_path, image))
    # input()
    shutil.copy(os.path.join(origin_path, image), os.path.join(target_path, str(id) + '.jpg'))
    id += 1

"""
# 重命名正样本
for image in positive_image_list:
    print(image)
    if image.startswith('B'):
        
        input()
        os.rename(os.path.join(positive_path, image), os.path.join(positive_path, str(id) + '.jpg' ))
        id += 1  

# 拷贝正样本
for image in bing_image_list:
    shutil.move(os.path.join(bing_path, image), os.path.join(positive_path, str(id) + '.jpg'))
    # input()
    # print(os.path.join(bing_path, image))
    id += 1
"""


