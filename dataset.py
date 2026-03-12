import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

# 1、 自定义的 Dataset 类

# =================================================================================================================================================
#    学习记录
#   这个类是继承的 PyTorch 的 Dataset 类，主要用于加载和处理手写数字数据集。它的主要功能包括：
#   1. 初始化方法 (__init__)：
#           接受数据集的根目录和可选的图像变换。它会遍历根目录下的每个数字文件夹（0-9），并将每个图像的路径和对应的标签存储在列表中。
#           相当于设定数据集的基础参数，并且找到初始数据，打好标签，为后边的训练做好准备。
#   2. 获取数据集长度方法 (__len__)：
#          其实就是输出这个数据集有多大
#   3. 获取单个数据项方法 (__getitem__)：
#           根据索引获取图像路径，加载图像并转换为灰度图，然后返回图像和对应的标签。如果提供了图像变换，则应用变换。
#           每次需要图片的时候，都会调用这个方法，它决定了数据的处理
# =================================================================================================================================================
class MyDigitsDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label in range(10):
            folder_path = os.path.join(root_dir,str(label))
            
            if os.path.isdir(folder_path):
                for img_name in os.listdir(folder_path): 
                    if img_name.endswith('.png'): 
                        self.image_paths.append(os.path.join(folder_path, img_name))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idex):
        img_path = self.image_paths[idex]
        image = Image.open(img_path).convert('L')  # 转换为灰度图
        label = self.labels[idex]

        if self.transform:
            image = self.transform(image)

        return image,label
    
# 2、 数据预处理和加载
# =================================================================================================================================================
#    学习记录
#    这个函数主要用于创建训练、数据加载器。它的主要功能包括：
#    1. 定义图像变换：
#           包括调整图像大小、反转（因为数据集中全部为白底黑字，进行颜色反转后更利好训练，此处原因后续会写）、
#           转换为张量和归一化（此处还没完全理解，目前认为是让数据在0附近震荡，模型就不用处理更多的数学特征，从而便于运算？）。这些变换有助于增强数据并使其适合模型训练。
#    2. 创建完整数据集：
#           使用上面自定义的类创建完整的数据集实例。
#    3. 划分训练集和验证集：
#           根据指定的训练比例，将完整数据集划分为训练集和验证集。
#    4. 创建数据加载器：
#           使用 DataLoader 类创建训练和验证数据加载器，设置批量大小、是否打乱数据以及工作线程数。
#    5. 返回数据加载器：
#           返回训练和验证数据加载器，以供模型训练和评估使用。
#
#个人认为黑底白字更适合，因为在灰度图中，黑色的值是0，白色的值是255，后续在进行矩阵乘法的时候，
#                      如果白底黑字，那大部分运算都用于背景，而黑底白字则更多的运算集中在数字部分，这样可能更有利于模型学习数字的特征。
# =================================================================================================================================================
def get_dataloaders(data_dir='my_digits', batch_size=32, train_ratio=0.8):

    transform_ = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.RandomInvert(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    full_dataset = MyDigitsDataset(root_dir = data_dir, transform = transform_)

    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size]) #随机打乱

    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 0
    )

    val_loader = DataLoader(
        dataset = val_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 0
    )

    return train_loader, val_loader


# 3、 测试数据加载器

if __name__ == "__main__":
    print("正在测试数据加载器...")
    train_dl, val_dl = get_dataloaders(data_dir='my_digits', batch_size=16)
    
    print(f"训练集批次数量: {len(train_dl)} (每批 16 张图)") # 此处len(train_dl) 输出的是批次数量
    print(f"验证集批次数量: {len(val_dl)}")
    
    for images, labels in train_dl:
        print(f"\n提取第一个Batch")
        print(f"图像张量形状 (Batch, Channel, Height, Width): {images.shape}") 
        print(f"标签张量形状 (Batch): {labels.shape}") 
        print(f"当前批次的真实标签: {labels}")
        break 