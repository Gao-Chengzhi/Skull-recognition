import os
import sys
sys.path.append('..')
# print(sys.path)
import torch
import time
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import argparse
from dinov2_classification_model import skull_model
import torch.optim as optim
import logging
import csv

parser = argparse.ArgumentParser(description='Fine-tune DINOv2 on skull dataset')
parser.add_argument('--log-dir', default='./log_dir', type=str, metavar='PATH',
                    help='path to directory where to log (default: current directory)')
parser.add_argument('--data-dir', default='../data/skull/Skull_Infringement_test', type=str, metavar='PATH',
                    help='path to the dataset')
parser.add_argument('--backbone-path', default='./model_weight/dinov2_vitg14_pretrain.pth', type=str, metavar='PATH',
                    help='path to the backbone checkpoint (default with reg)')

args = parser.parse_args()
print(args)

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define transforms for the training and validation datasets
data_transforms = {
    'train': transforms.Compose([
        # 调整scale参数 scale=(0.08, 1.0) 裁剪比例上下界
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

test_dataset = datasets.ImageFolder(args.data_dir, data_transforms['val'])
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(args.log_dir, 'epoch0_model.pth')
model = skull_model(class_num=1, backbone_path=args.backbone_path).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# 测试集预测
image_paths = []
predictions = []
# 记录开始时间
start_time = time.time()
for batch_idx, (inputs, _) in enumerate(tqdm(test_dataloader)):
    inputs = inputs.to(device)
    start_idx = batch_idx * test_dataloader.batch_size
    end_idx = start_idx + inputs.size(0)
    # 获取当前 batch 中图片的路径
    batch_paths = [test_dataloader.dataset.samples[idx][0] for idx in range(start_idx, end_idx)]    
    print(batch_paths[0])
    image_paths.extend([path.split('/')[5] for path in batch_paths])
    with torch.no_grad():
        features = model.feature_model(inputs)
        # outputs = model.classifier(features)
        outputs = model.classifier(features).squeeze()

        probabilities = torch.sigmoid(outputs)

        preds = (probabilities >= 0.55).long()
        # _, preds = torch.max(outputs, 1)
    
    predictions.extend(preds.cpu().numpy())

print(len(predictions),len(image_paths))

end_time = time.time()
elapsed_time = end_time - start_time
print(f"完成测试，耗时: {elapsed_time:.2f} 秒")
# 准备 CSV 文件
csv_file_path = os.path.join(args.log_dir, 'test.csv')
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'prediction'])

    # 将图片路径和预测结果写入 CSV 文件
    for path, pred in zip(image_paths, predictions):
        writer.writerow([path, pred])
