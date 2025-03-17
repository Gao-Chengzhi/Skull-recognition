import gradio as gr
import torch
import torchvision.transforms as transforms
import sys
sys.path.append('..')

from dinov2_classification_model import skull_model

model_path = "./log_dir/epoch0_model.pth"
backbone_path = "./model_weight/dinov2_vitg14_pretrain.pth"
device = "cuda:0"
model = skull_model(class_num=1, backbone_path=backbone_path).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

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

def infer(image, score=0.55):

    input_tensor = data_transforms['val'](image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model.feature_model(input_tensor)
        output = model.classifier(features).squeeze()
        confidence = torch.sigmoid(output)
    
    prediction = int((confidence >= score).item())
    if prediction == 1:
        result = f"yes！检测存在骷髅头样式, score: {confidence} "
    else:
        result = f"no！检测不存在骷髅头样式, score: {confidence} "

    return result

demo = gr.Interface(
    fn=infer,
    inputs=[gr.Image(type="pil"), gr.Slider(0, 1, value=0.55, step=0.1, label="置信度阈值")],
    outputs="text",
    title="骷髅头样式检测",
    description="上传一张商品图像，模型将检测是否存在骷髅头样式"
)

demo.launch(server_name="0.0.0.0", server_port=7862)
