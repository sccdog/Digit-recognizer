import torch
import torch.nn as nn  # 神经网络 Neural network
import torch.nn.functional as F  # 函数 Function
from torch.utils.data import DataLoader
from DataAugmentation import train_data, valid_data, test_data
from Dataloader import train_sampler, valid_sampler
global model, device

test_dataloader = DataLoader(dataset=test_data)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.dr = nn.Dropout(p=0.4)
        self.conv1 = nn.Conv2d(1, 32, (3, 3))  # 输出大小转变为 26x26
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (3, 3))  # 输出大小转变为 24x24
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, (5, 5), stride=(2, 2))  # 输出大小转变为 10x10
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, (3, 3))  # 输出大小转变为 8x8
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, (3, 3))  # 输出大小转变为 6x6
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, (5, 5), stride=(2, 2))  # 输出大小转变为 1x1
        self.bn6 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 1 * 1, 128)  # 特征数 * 输出维度，全连接层神经元数量
        self.bn7 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):  # 正向传播
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.dr(self.bn3((F.relu(self.conv3(x)))))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))
        x = self.dr(self.bn6(F.relu(self.conv6(x))))
        x = x.view(-1, 64 * 1 * 1)
        x = self.dr(self.bn7(F.relu(self.fc1(x))))
        x = self.fc2(x)

        return x


def train(num_epochs, batch_size, learning_rate):
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, sampler=train_sampler)
    global model, device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 条件允许下使用CUDA进行训练
    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 随着训练的进行逐步降低学习率，以更好的找到局部最优结果
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    n_total_steps = len(train_dataloader)  # 用于每个训练时期结束后显示一次损失(loss)
    for epoch in range(num_epochs):
        model.train()  # 模块将以训练模式training mode运行
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            # 正向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播与优化
            optimizer.zero_grad()  # 将所有已被优化的Torch.Tensor的梯度设置为0
            loss.backward()  # 计算每个requires_grad=True的参数x的 d(loss)/dx
            optimizer.step()  # 使用梯度x.grad更新x的值。
            if (i + 1) % n_total_steps == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}, Loss = {loss.item():4f}]'
                )

        lr_scheduler.step()
        print('Current learning rate:', optimizer.param_groups[0]['lr'])
        get_valid_acc()

    return model


def get_valid_acc():
    valid_dataloader = DataLoader(dataset=valid_data, sampler=valid_sampler)
    # 因为验证不需要反向传播，我们暂时将tensor的required_grad属性设置为False，并停用计算梯度的Autograd引擎
    with torch.no_grad():
        # 模块将以评估模式evaluation mode运行
        # 忽略包括Dropout等在内的各层
        model.eval()
        n_correct = 0
        n_samples = 0
        for images, labels in valid_dataloader:
            images = images.reshape(-1, 1, 28, 28).to(device)
            labels = labels.to(device)
            outputs = model(images)

            predictions = outputs.cuda().data.max(1, keepdim=True)[1]
            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Validation set accuracy = {acc}')
