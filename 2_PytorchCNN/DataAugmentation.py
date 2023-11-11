from torchvision import transforms  # 图像数据预处理 Image preprocessing
from Dataloader import DatasetMNIST, train_set, test_set

# 训练集变换
transform_train = transforms.Compose([
    transforms.ToPILImage(),  # 将张量tensor或ndarray转换为PIL图像，允许RandomAffine对数据进行操作
    transforms.RandomAffine(degrees=10, translate=(0, 0.1), scale=(0.9, 1.1)),  # 随机旋转, 位移, 与缩放
    transforms.ToTensor(),  # 将PIL式格式化为可被pytorch快速处理的张量类型
    transforms.Normalize((0.1307,), (0.3081,))
    # 标准化数据以加速梯度下降的收敛
    # Mean Pixel Value / 255 = 33.31002426147461 / 255 = 0.1307
    # Pixel Values Std / 255 = 78.56748962402344 / 255 = 0.3081
])

# 验证集变换
transform_valid = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 测试集变换
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 创建训练集、验证集、测试集
train_data = DatasetMNIST(train_set, transform=transform_train)
valid_data = DatasetMNIST(train_set, transform=transform_valid)
test_data = DatasetMNIST(test_set, transform=transform_test, labeled=False)
