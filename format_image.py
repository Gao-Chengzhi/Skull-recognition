
import os
from PIL import Image
import warnings
folder_path = '../data/skull/Skull_Infringement/postive'
extensions = []
index=0


for filee in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filee)
    # print('** Path: {}  **'.format(file_path), end="\r", flush=True)
    # print(file_path)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            im = Image.open(file_path)
            
            rgb_im = im.convert('RGB')
            if len(w) > 0:
                for warning in w:
                    if "unclosed file" not in str(warning.message):
                        print(f"UserWarning: {w[-1].message} | Image Path: {file_path}")
                        # 删除RGBA文件
                        os.remove(file_path)
    except Exception as e:
        print(f"Error: {e} | Image Path: {file_path}")
    # input()
