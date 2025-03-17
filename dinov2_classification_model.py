from dinov2.hub.backbones import dinov2_vitl14_reg, dinov2_vitg14
from dinov2.eval.utils import ModelWithIntermediateLayers
import torch.nn as nn
import torch
from functools import partial
from dinov2.eval.linear import LinearClassifier, MultiLayerClassifier, MultiLayer4Classifier
from dinov2.eval.linear import create_linear_input
import torchvision.transforms as transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class skull_model(nn.Module):
    def __init__(self, class_num, backbone_path):
        super(skull_model, self).__init__()
        # print(backbone_path)
        # input()
        # self.dinov2_backbone = dinov2_vitl14_reg(weights={'LVD142M':backbone_path})
        self.dinov2_backbone = dinov2_vitg14(weights={'LVD142M':backbone_path})
        # 调整精度
        autocast_ctx = partial(
            torch.cuda.amp.autocast, enabled=True, dtype=torch.float32
        )
       
        # n_last_blocks选择倒数第几个层的特征
        self.feature_model = ModelWithIntermediateLayers(
            self.dinov2_backbone, n_last_blocks=1, autocast_ctx=autocast_ctx
        ).to(device)

        with torch.no_grad():
            sample_input = torch.randn(1, 3, 224, 224).to(device)
            # print(type(sample_input))
            sample_output = self.feature_model(sample_input)
        # print(f'sample_output.shape{sample_output[0][0].shape, sample_output[0][1].shape}')
        # input()
        # use_avgpool 表示是否将patch tokens的信息并入class token中
        out_dim = create_linear_input(
            sample_output, use_n_blocks=1, use_avgpool=True
        ).shape[1]
        print(f'output dim {out_dim}')
        """self.classifier = MultiLayerClassifier(
            out_dim, hidden_dim=1024, use_n_blocks=1, use_avgpool=True, num_classes=class_num
        ).to(device)
        self.classifier = LinearClassifier(
            out_dim, use_n_blocks=1, use_avgpool=True, num_classes=class_num
        ).to(device)
        self.classifier = MultiLayer4Classifier(
            out_dim, hidden_dim1=1024, hidden_dim2=512, hidden_dim3=128, use_n_blocks=1, use_avgpool=True, num_classes=class_num
        ).to(device)"""
        self.classifier = MultiLayerClassifier(
            out_dim, hidden_dim=1024, use_n_blocks=1, use_avgpool=True, num_classes=class_num
        ).to(device)
    def forward(self, x):
        x = self.feature_model(x)
        x = self.classifier(x)
        return x

import torchvision.models as models
import torch.nn as nn
import torch

class ResNetSkullModel(nn.Module):
    def __init__(self, class_num):
        super(ResNetSkullModel, self).__init__()
        # 使用预训练的 ResNet50
        self.backbone = models.resnet50(weights='IMAGENET1K_V1')  # 可根据需要更换版本
        # 修改最后的全连接层以匹配分类数
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, class_num)

    def forward(self, x):
        x = self.backbone(x)  # 使用 ResNet
        return x


"""
from transformers import AutoModel, CLIPImageProcessor
import torch.nn as nn
import torch

# internVL

class InternViT(nn.Module):
    def __init__(self, class_num):
        super(InternViT, self).__init__()

        self.internvit_backbone = AutoModel.from_pretrained(
            '/home/xray/gcz/dinov2/skull_classification/model_weight/InternViT-300M-448px',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).cuda().eval() 

        self.classifier = nn.Linear(1024, class_num)

    def forward(self, x):
        # 前向传播特征提取
        with torch.no_grad():
            features = self.internvit_backbone(x).pooler_output

        # 分类头
        x = self.classifier(features.to(torch.float32))  # 取第一个 token
        return x
"""