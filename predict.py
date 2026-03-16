import torch
from model import SimpleCNN
from PIL import Image
import torchvision.transforms as transforms
import os

# ==============================================================================
# 模型预测
# 功能：
# 1. 载入先前已训练好并保存的最佳模型权重 (best_model.pth)
# 2. 接收一张全新的单手写数字图片
# 3. 对该图片进行预测，输出模型认为它是什么数字，以及预测的置信度
# ==============================================================================

def load_trained_model(model_path="best_model.pth", device=None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 实例化一个结构一模一样的空模型
    model = SimpleCNN().to(device)
    
    # 检查权重文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型权重文件: {model_path}。请先运行 train.py 训练模型。")
    
    # 将保存的权重字典 (state_dict) 加载到模型中
    # map_location 用于处理：如果在带有 GPU 的机器上训练保存，但在仅有 CPU 的机器上加载时，防止报错
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    # 切换为评估模式
    model.eval()
    return model

def predict_single_image(model, image_path, device=None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # 1. 加载并转换为灰度图
        image = Image.open(image_path).convert('L')
    except FileNotFoundError:
        print(f"找不到要预测的图片文件: {image_path}，请检查路径。")
        return None, None

    # 2. 定义预处理流程 (必须与 dataset.py 中的 transform 保持高度一致)
    # 训练时的 transform_ 是应对整个 batch 的，这里我们也需要对单图做一样的事
    preprocess = transforms.Compose([
        transforms.Resize((28, 28)),          # 统一缩放至 28x28
        transforms.RandomInvert(p=1.0),       # 反转颜色：让白底黑字变为黑底白字，利于提取特征
        transforms.ToTensor(),                # 转换为 PyTorch 张量，且像素值从 0-255 变成 0.0-1.0
        transforms.Normalize((0.5), (0.5))    # 归一化 (减去0.5，除以0.5，使得数据围绕 0 分布)
    ])

    # 对图片进行变换：经过处理后，张量的 shape 是 [1, 28, 28] -> [Channels, Height, Width]
    input_tensor = preprocess(image)
    
    # 模型接收的必须是有批量维度的 (Batch Size)
    # 所以要在第 0 维增加一维，让它变成 [1, 1, 28, 28] 分别对应 [Batch, Channel, Height, Width]
    input_batch = input_tensor.unsqueeze(0).to(device)

    # 3. 使用模型进行预测
    with torch.no_grad():
        output = model(input_batch)
        
        # 将无界的原始得分 (Logits) 转化为标准的概率分布 (加和为 1.0 的 0~1 的值)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # 找到概率最大的那个分类：概率值本身 和 该分类对应的索（即真实数字0-9）
        max_prob, predicted_digit = torch.max(probabilities, 0)

    # 提取纯粹的 Python 标量值
    predicted_digit = predicted_digit.item()
    confidence = max_prob.item() * 100 # 换算成百分比

    return predicted_digit, confidence

if __name__ == "__main__":
    
    # 1. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        print("正在加载模型...")
        # 实例化模型并读入最佳权重
        model = load_trained_model("best_model.pth", device=device)
        print("模型加载成功！")
        
        # 2. 准备一张测试图片
        test_image_path = "my_digits/0/img_0_111.png"
        
        print(f"\n准备对图片进行预测: {test_image_path}")
        
        # 3. 进行推理
        predicted_digit, conf = predict_single_image(model, test_image_path, device=device)
        
        if predicted_digit is not None:
            print("=" * 40)
            print("                预测结果                ")
            print("=" * 40)
            print(f"模型认为这张图片上的数字是: \033[1;32m{predicted_digit}\033[0m") # \033[... 用来将终端文字打印为绿色
            print(f"模型的自信程度 (置信度): {conf:.2f}%")
            print("=" * 40)
            
    except Exception as e:
        print(f"\n程序运行出错: {e}")
