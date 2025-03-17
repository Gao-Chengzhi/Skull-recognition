import shutil
import random
import os
from tqdm import tqdm
# bing_path = '/home/xray/gcz/data/skull/Skull_Infringement/object_with_skull'
# positive_path = '/home/xray/gcz/data/skull/Skull_Infringement/postive'
# trade_image_path = '/home/xray/gcz/data/trademark/Trademark_Smart_Match/images'
target_image_path = '../data/skull/Skull_Infringement/negtive'
logo_image_path = '../data/logo_test_data'
# trade_image_list = os.listdir(trade_image_path)
logo_image_list = os.listdir(logo_image_path)
# bing_image_list = os.listdir(bing_path)
# positive_image_list = os.listdir(positive_path)
# print(origin_image_list)
# 9141
negtive_sample_list = tqdm(random.sample(logo_image_list, 9141))
# coco_sample_list = tqdm(random.sample(coco_image_list, 2910))
id = len(os.listdir(target_image_path))
print(id)
#input()
# 拷贝负样本
for image in negtive_sample_list:
    # print(os.path.join(trade_image_path, image))
    # input()
    shutil.copy(os.path.join(logo_image_path, image), os.path.join(target_image_path, str(id) + '.jpg'))
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


