import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time
import gc

# 1. 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. 数据集下载和预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 标准化处理
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 3. 定义卷积神经网络模型（包含4层卷积层）
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 4. 初始化模型、损失函数和优化器
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 创建目录保存结果
os.makedirs('results', exist_ok=True)
timestamp = time.strftime("%Y%m%d-%H%M%S")
results_dir = f'results/run_{timestamp}'
os.makedirs(results_dir, exist_ok=True)

# 使用固定批量大小
batch_size = 256
print(f"使用批量大小: {batch_size}")

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# 5. 训练模型 (32轮)
def train_model():
    num_epochs = 32  # 32轮训练
    
    # 初始化记录器
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epoch_time': [],
    }
    
    best_test_acc = 0.0  # 用于保存最佳模型
    
    # 学习率调度器   
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 训练阶段
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 计算训练指标
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        
        # 测试阶段
        test_loss, test_accuracy = evaluate_model()
        
        # 更新学习率
        scheduler.step(test_accuracy)
        
        # 记录时间
        epoch_time = time.time() - start_time
        
        # 打印结果
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%, '
              f'Time: {epoch_time:.2f}s')
        
        # 保存历史记录
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_accuracy)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_accuracy)
        history['epoch_time'].append(epoch_time)
        
        # 保存最佳模型
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            torch.save(model.state_dict(), f'{results_dir}/best_model.pth')
            print(f"Saved best model with test accuracy: {test_accuracy:.2f}%")
    
    # 保存训练历史为CSV
    df = pd.DataFrame(history)
    df.to_csv(f'{results_dir}/training_history.csv', index=False)
    print(f"Training history saved to {results_dir}/training_history.csv")
    
    return history

# 6. 评估模型
def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100 * correct / total
    return test_loss, accuracy

# 7. 保存模型权重
def save_model():
    torch.save(model.state_dict(), f'{results_dir}/final_model.pth')
    print(f"Final model saved as {results_dir}/final_model.pth")

# 8. 绘制训练曲线
def plot_training_history(history):
    plt.figure(figsize=(14, 10))
    
    # 绘制损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
    plt.plot(history['epoch'], history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(history['epoch'], history['train_acc'], label='Train Accuracy')
    plt.plot(history['epoch'], history['test_acc'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 绘制训练时间曲线
    plt.subplot(2, 2, 3)
    plt.plot(history['epoch'], history['epoch_time'], 'g-')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.title('Epoch Training Time')
    plt.grid(True)
    
    # 移除显存使用曲线部分
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/training_curves.png', dpi=300)
    plt.close()
    print(f"Training curves saved to {results_dir}/training_curves.png")
    
    # 单独绘制损失曲线
    plt.figure(figsize=(8, 5))
    plt.plot(history['epoch'], history['train_loss'], 'b-', label='Train Loss')
    plt.plot(history['epoch'], history['test_loss'], 'r-', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{results_dir}/loss_curves.png', dpi=300)
    plt.close()
    
    # 单独绘制准确率曲线
    plt.figure(figsize=(8, 5))
    plt.plot(history['epoch'], history['train_acc'], 'b-', label='Train Accuracy')
    plt.plot(history['epoch'], history['test_acc'], 'r-', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{results_dir}/accuracy_curves.png', dpi=300)
    plt.close()

# 9. 主函数
def main():
    # 清除缓存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 训练模型
    history = train_model()
    
    # 绘制曲线
    plot_training_history(history)
    
    # 保存最终模型
    save_model()
    
    # 最终清理
    torch.cuda.empty_cache()
    gc.collect()
    
    # 打印最终结果
    best_epoch = history['test_acc'].index(max(history['test_acc'])) + 1
    print(f"\n训练完成! 最佳模型在 epoch {best_epoch}, 测试准确率: {max(history['test_acc']):.2f}%")

if __name__ == "__main__":
    main()