import os
import sys
sys.path.append('..')
# print(sys.path)
import torch
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
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import MulticlassF1Score, BinaryF1Score
import csv
parser = argparse.ArgumentParser(description='Fine-tune DINOv2 on skull dataset')
parser.add_argument('--batch-size', '-b', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--log-dir', default='./log_dir', type=str, metavar='PATH',
                    help='path to directory where to log (default: current directory)')
parser.add_argument('--data-dir', default='../data/skull/Skull_Infringement', type=str, metavar='PATH',
                    help='path to the dataset')
parser.add_argument('--backbone-path', default='./model_weight/dinov2_vitg14_pretrain.pth', type=str, metavar='PATH',
                    help='path to the backbone checkpoint ')

args = parser.parse_args()
print(args)
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()
# 设置日志文件

log_file = os.path.join(args.log_dir, 'training.log')
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logger.addHandler(file_handler)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define transforms for the training and validation datasets
data_transforms = {
    'train': transforms.Compose([
        # 调整scale参数 scale=(0.08, 1.0) 裁剪比例上下界
        transforms.Lambda(lambda image: image.convert("RGB")),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Lambda(lambda image: image.convert("RGB")),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

test_dataset = datasets.ImageFolder('../data/skull/Skull_Infringement_test', data_transforms['val'])
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


image_dataset = datasets.ImageFolder(args.data_dir, data_transforms['train'])
# train_dataloader = torch.utils.data.DataLoader(image_train_dataset, batch_size=32, shuffle=True, num_workers=4)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataloader = DataLoader(image_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

print(f'train data num {len(train_dataloader) * args.batch_size}')
model = skull_model(class_num=1, backbone_path=args.backbone_path).to(device)
# Disable gradient for feature model
for param in model.feature_model.parameters():
    param.requires_grad = False
    
for param in model.classifier.parameters():
    param.requires_grad = True


# Define loss function
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
# Observe that all parameters are being optimized
optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# f1_metric = MulticlassF1Score(num_classes=2).to(device)
f1_metric = BinaryF1Score().to(device)
# Train the model
num_epochs = 1
best_acc = 0.0
for epoch in range(num_epochs):

    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    
    start_time = time.time()

    model.feature_model.eval()
    model.classifier.train()


    running_loss = 0.0
    running_corrects = 0

    batch_counter = 0
    for inputs, labels in tqdm(train_dataloader):
        """
        if batch_counter >= 5:
            break
        batch_counter += 1
        """
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        with torch.no_grad():
            features = model.feature_model(inputs)

        with torch.set_grad_enabled(True):
            outputs = model.classifier(features).squeeze()
            # print(outputs.shape)
            #input()
            # _, preds = torch.max(outputs, 1)
            probabilities = torch.sigmoid(outputs)

            preds = (probabilities >= 0.55).long()
            # preds = (outputs >= 0.2).long()
            loss = criterion(outputs, labels.float())

            
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataloader.dataset)
    epoch_acc = running_corrects.double() / len(train_dataloader.dataset)
    logger.info(f'Epoch {epoch}/{num_epochs - 1} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    # Save the best model
    # if val_acc > best_acc:
    # best_acc = val_acc
    save_path = os.path.join(args.log_dir, f'epoch{epoch}_model.pth')
    torch.save(model.state_dict(), save_path)
    logger.info(f'Model saved at {save_path}')
    
    """
    if epoch == num_epochs - 1:
        save_path = os.path.join(args.log_dir, 'last_model.pth')
        torch.save(model.state_dict(), save_path)
        logger.info(f'last Model saved at {save_path}')
    """

    # test 
    model.eval()

        # 测试集预测
    image_paths = []
    predictions = []

    for batch_idx, (inputs, _) in enumerate(tqdm(test_dataloader)):
        inputs = inputs.to(device)
        start_idx = batch_idx * test_dataloader.batch_size
        end_idx = start_idx + inputs.size(0)
        # 获取当前 batch 中图片的路径
        batch_paths = [test_dataloader.dataset.samples[idx][0] for idx in range(start_idx, end_idx)]    
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
    
    # 准备 CSV 文件
    csv_file_path = os.path.join(args.log_dir, f'test{epoch}.csv')
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'prediction'])

        # 将图片路径和预测结果写入 CSV 文件
        for path, pred in zip(image_paths, predictions):
            writer.writerow([path, pred])

logger.info('Training complete')