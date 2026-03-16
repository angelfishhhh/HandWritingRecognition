import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    # 所有自定义的模型都必须继承 PyTorch 的 nn.Module 父类
    
    def __init__(self):
        super(SimpleCNN, self).__init__() # 调用父类的初始化逻辑，python固定语法
        
        # 1. 第一层卷积提取特征
        # in_channels=1: 因为传入的是黑白灰度图，只有 1 个颜色通道
        # out_channels=16: 我们使用 16 个不同的卷积核，提取 16 种不同的特征图
        # kernel_size=3: 卷积核的大小是 3x3
        # padding=1: 在图片边缘填充一圈 0，保证卷积后图片长宽不缩小 (维持 28x28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        
        # 2. 第二层卷积提取深层特征
        # 接收上一层传来的 16 个特征通道，输出 32 个更高级的特征通道
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # 3. 池化层
        # 2x2 的窗口，每次移动 2 步。这会把特征图的宽和高都缩小一半
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 4. 全连接层 (分类)
        # 经过两次池化，原始 28x28 的图片变成了 7x7 (28 -> 14 -> 7)
        # 我们有 32 个特征通道，所以展平后的总神经元个数是 32 * 7 * 7
        self.fc1 = nn.Linear(32 * 7 * 7, 128) # 将这些特征映射到 128 个隐藏神经元
        
        # 5. Dropout层 (防止过拟合)
        # 训练时随机让50%的神经元“失活”（输出变为0）
        self.dropout = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(128, 10)         # 最终输出 10 个神经元，对应 0-9 这 10 个类别

    def forward(self, x):

        #=====================================
        # 激活函数不容易理解
        # 目前我认为它就是一种数学公式，用来增加非线性的""能力"
        # 具体的公式是：f(x) = max(0, x)
        # 也就是：如果 x > 0，则 f(x) = x；如果 x <= 0，则 f(x) = 0
        # 这样的话，如果x<0，它就会被置为0，也就是出现了一个拐角，不断训练下，模型就能近似出拐角或曲线
        #=====================================


        # 第一步：输入 x -> 卷积1 -> ReLU激活 -> 池化
        # 输入尺寸: [Batch, 1, 28, 28] -> 卷积后: [Batch, 16, 28, 28] -> 池化后: [Batch, 16, 14, 14]
        x = self.pool(F.relu(self.conv1(x)))
        
        # 第二步：继续卷积 -> ReLU激活 -> 池化
        # 输入尺寸: [Batch, 16, 14, 14] -> 卷积后: [Batch, 32, 14, 14] -> 池化后: [Batch, 32, 7, 7]
        x = self.pool(F.relu(self.conv2(x)))
        
        # 第三步：展平
        # 使用 view 函数改变张量的形状。-1 代表让 PyTorch 自动推导批次大小 (Batch Size)
        # 尺寸变化: [Batch, 32, 7, 7] -> [Batch, 1568]
        x = x.view(-1, 32 * 7 * 7)
        
        # 第四阶段：全连接分类
        x = F.relu(self.fc1(x)) # 通过第一个全连接层，并使用 ReLU 激活函数增加非线性
        x = self.dropout(x)     # 加入 Dropout 随机失活
        x = self.fc2(x)         # 通过输出层。注意：这里不需要激活函数，原始的得分 (Logits) 会直接交给后续的损失函数处理
        
        return x

# 测试部分

if __name__ == '__main__':
    model = SimpleCNN()
    print("模型结构：\n", model)
    
    dummy_input = torch.randn(4, 1, 28, 28) #假装输入4张28x28的黑白图片
    
    output = model(dummy_input)
    print(f"\n模拟输入张量形状: {dummy_input.shape}")
    print(f"模型输出张量形状: {output.shape} (预期为 [4, 10])")