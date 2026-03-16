import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleCNN
from dataset import get_dataloaders
import os
from tqdm import tqdm  #用于显示进度条

# ==============================================================================
# 模型训练
# 1. 加载数据集
# 2. 初始化cnn
# 3. 定义损失函数和优化器
# 4. 执行训练循环并实时验证
# 5. 保存准确率最高的模型权重
# ==============================================================================

def train_model(data_dir='my_digits', epochs=20, batch_size=32, learning_rate=0.001):
    # 1. 设置设备 (如果有显卡/CUDA则使用GPU加速，否则使用CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用的计算设备: {device}")

    # 2. 获取数据加载器
    print("正在加载数据...")
    train_loader, val_loader = get_dataloaders(data_dir=data_dir, batch_size=batch_size)
    print(f"数据加载完成！训练批次: {len(train_loader)}，验证批次: {len(val_loader)}")

    # 3. 初始化模型，并将其放入计算设备中
    model = SimpleCNN().to(device)

    # 4. 定义损失函数 (交叉熵损失)
    # 它会自动对模型输出的非标准分数进行 Softmax 处理，所以我们在模型定义里没有加 Softmax
    # loss越大，得分越低(对于相应的概率取对数并加负号)
    criterion = nn.CrossEntropyLoss()

    # 5. 定义优化器 
    # Adam优化器，解决了困在局部最优解的问题，还有自适应学习率
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_accuracy = 0.0 # 用于记录最高的验证集准确率
    best_model_path = "best_model.pth"

    # 6. 开始训练循环
    print("开始训练...")
    for epoch in range(epochs):

        model.train() # 将模型设置为训练模式 (启用 Dropout/BatchNorm 等)
        running_loss = 0.0 # 记录当前 Epoch 的累计损失
        # 使用 tqdm 包装 train_loader 来显示带有描述的进度条
        # leave=False 表示当前 epoch 结束后清除这行进度条，保持终端整洁
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] 训练中", leave=False)
        
        for images, labels in train_bar:
            # 将数据和标签移动到同一设备上
            images, labels = images.to(device), labels.to(device)

            # 步骤 1：梯度清零
            optimizer.zero_grad()

            # 步骤 2：前向传播 (将图片送入模型得到预测结果)
            outputs = model(images)

            # 步骤 3：计算损失 (对比预测结果与真实标签)
            loss = criterion(outputs, labels)

            # 步骤 4：反向传播 (计算模型每个参数对误差的影响 -> 梯度求导)
            loss.backward()

            # 步骤 5：更新模型参数
            optimizer.step()

            # 累加批次损失
            running_loss += loss.item()
            
            # 实时更新进度条后缀信息（当前批次的损失）
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 计算当前 Epoch 的平均训练损失
        avg_train_loss = running_loss / len(train_loader)

        model.eval() # 将模型设置为评估模式
        val_loss = 0.0
        correct = 0 # 记录预测正确的样本数量
        total = 0   # 记录验证集总样本数量

        
        # 使用 tqdm 包装 val_loader
        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] 验证中", leave=False)

        # 使用 torch.no_grad() 可以节省内存并加快计算速度，因为不需要准备反向传播的信息
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                
                # 前向传播预测
                outputs = model(images)
                
                # 累计验证损失
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # 获取预测的类别 (找出模型输出的10个得分中最大的那个下标)
                # torch.max 返回两个值：最大值(下划线) 和 最大值对应的索引(即预测标签)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0) # 累加批次大小
                correct += (predicted == labels).sum().item()

        # 计算当前 Epoch 的平均验证损失和准确率
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        # 每个 Epoch 结束后打印一次结果
        print(f"Epoch [{epoch+1}/{epochs}] | 训练损失: {avg_train_loss:.4f} | "
              f"验证损失: {avg_val_loss:.4f} | 验证准确率: {val_accuracy:.2f}%")

        # 保存表现最好的模型权重
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # torch.save 函数用于保存模型的参数字典 (state_dict)
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> 发现更好的模型！准确率提升至 {best_val_accuracy:.2f}%，已保存至 {best_model_path}")

    print(f"训练全部完成！最高验证集准确率为: {best_val_accuracy:.2f}%")
    print(f"最佳模型已保存在当前目录下：{best_model_path}")

if __name__ == "__main__":
    import os
    # 动态获取数据集路径
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    data_dir = os.path.join(current_dir, 'my_digits')
    
    # 检查数据集是否存在
    if os.path.exists(data_dir):
        train_model(data_dir=data_dir, epochs=15, batch_size=32, learning_rate=0.001)
    else:
        print(f"错误: 找不到数据集目录 '{data_dir}'。")
