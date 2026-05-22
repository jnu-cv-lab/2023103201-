import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # 解决Windows卡顿

# ======================
# 设备 & 数据（MNIST）
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# 提速关键：batch_size 变大
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

# ======================
# CNN 模型（你的原模型，完全不变）
# ======================
class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ======================
# 任务 1：训练基准模型
# ======================
model = MNIST_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5

train_loss = []
train_acc = []
test_acc = []

for epoch in range(epochs):
    model.train()
    tl, ta = 0.0, 0
    for img, lab in train_loader:
        img, lab = img.to(device), lab.to(device)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, lab)
        loss.backward()
        optimizer.step()
        tl += loss.item()
        ta += (out.argmax(1) == lab).sum().item()

    tl /= len(train_loader)
    ta = 100 * ta / len(train_dataset)
    train_loss.append(tl)
    train_acc.append(ta)

    model.eval()
    te = 0
    with torch.no_grad():
        for img, lab in test_loader:
            img, lab = img.to(device), lab.to(device)
            te += (model(img).argmax(1) == lab).sum().item()
    te = 100 * te / len(test_dataset)
    test_acc.append(te)

    print(f"任务1 - Epoch {epoch+1} | Loss: {tl:.4f} | Train Acc: {ta:.2f}% | Test Acc: {te:.2f}%")

plt.figure(figsize=(10,4))
plt.subplot(121)
plt.plot(train_loss, label="Train Loss")
plt.title("Task1 Loss")
plt.legend()

plt.subplot(122)
plt.plot(train_acc, label="Train Acc")
plt.plot(test_acc, label="Test Acc")
plt.title("Task1 Accuracy")
plt.legend()
plt.savefig("task1.png")

# ======================
# 任务 2：优化器对比（SGD / Momentum / Adam）
# ======================
def run(opt_name, lr):
    model = MNIST_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    if opt_name == "SGD":
        opt = optim.SGD(model.parameters(), lr=lr)
    elif opt_name == "Momentum":
        opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name == "Adam":
        opt = optim.Adam(model.parameters(), lr=lr)
    else:
        opt = optim.Adam(model.parameters(), lr=lr)

    loss_hist = []
    acc_hist = []
    print(f"\n===== {opt_name} =====")
    for epoch in range(5):
        model.train()
        ls, ac = 0, 0
        for img, lab in train_loader:
            img, lab = img.to(device), lab.to(device)
            opt.zero_grad()
            out = model(img)
            loss = criterion(out, lab)
            loss.backward()
            opt.step()
            ls += loss.item()
            ac += (out.argmax(1) == lab).sum().item()
        ls /= len(train_loader)
        ac = 100 * ac / len(train_dataset)
        loss_hist.append(ls)
        acc_hist.append(ac)
        print(f"Epoch {epoch+1} | Loss: {ls:.4f} | Acc: {ac:.2f}%")

    # 最终测试
    model.eval()
    cor = 0
    with torch.no_grad():
        for img, lab in test_loader:
            img, lab = img.to(device), lab.to(device)
            cor += (model(img).argmax(1) == lab).sum().item()
    final = 100 * cor / len(test_dataset)
    print(f"[{opt_name}] 测试集准确率: {final:.2f}%")
    return loss_hist, acc_hist, final

# 运行三种优化器
loss_sgd, acc_sgd, test_sgd = run("SGD", 0.01)
loss_mom, acc_mom, test_mom = run("Momentum", 0.01)
loss_adam, acc_adam, test_adam = run("Adam", 0.001)

# 绘图
plt.figure(figsize=(12,5))
plt.subplot(121)
plt.plot(loss_sgd, label="SGD")
plt.plot(loss_mom, label="Momentum")
plt.plot(loss_adam, label="Adam")
plt.title("Task2 Loss Comparison")
plt.legend()

plt.subplot(122)
plt.plot(acc_sgd, label="SGD")
plt.plot(acc_mom, label="Momentum")
plt.plot(acc_adam, label="Adam")
plt.title("Task2 Accuracy Comparison")
plt.legend()
plt.savefig("task2.png")


# ======================
# 任务 3：学习率对比（固定Adam，lr=0.1/0.01/0.001）
# ======================
def run_with_lr(lr):
    model = MNIST_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_hist = []
    acc_hist = []
    print(f"\n===== 学习率: {lr} =====")
    for epoch in range(5):
        model.train()
        ls, ac = 0, 0
        for img, lab in train_loader:
            img, lab = img.to(device), lab.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out, lab)
            loss.backward()
            optimizer.step()

            ls += loss.item()
            ac += (out.argmax(1) == lab).sum().item()

        ls /= len(train_loader)
        ac = 100 * ac / len(train_dataset)
        loss_hist.append(ls)
        acc_hist.append(ac)
        print(f"Epoch {epoch+1} | Loss: {ls:.4f} | Acc: {ac:.2f}%")

    # 测试准确率
    model.eval()
    cor = 0
    with torch.no_grad():
        for img, lab in test_loader:
            img, lab = img.to(device), lab.to(device)
            cor += (model(img).argmax(1) == lab).sum().item()
    final = 100 * cor / len(test_dataset)
    print(f"测试集准确率: {final:.2f}%\n")
    return loss_hist, acc_hist, final

# 运行三种学习率（只跑一次）
lr_list = [0.1, 0.01, 0.001]
loss_lr = []
acc_lr = []
test_acc_lr = []

for lr in lr_list:
    l, a, te = run_with_lr(lr)
    loss_lr.append(l)
    acc_lr.append(a)
    test_acc_lr.append(te)

# 绘制对比曲线
plt.figure(figsize=(12,5))
plt.subplot(121)
for i, lr in enumerate(lr_list):
    plt.plot(loss_lr[i], label=f"lr={lr}")
plt.title("Task3 Loss (不同学习率对比)")
plt.legend()

plt.subplot(122)
for i, lr in enumerate(lr_list):
    plt.plot(acc_lr[i], label=f"lr={lr}")
plt.title("Task3 Accuracy (不同学习率对比)")
plt.legend()

plt.savefig("task3.png")

print("\n===== 三种学习率最终测试准确率 =====")
for i, lr in enumerate(lr_list):
    print(f"lr={lr} : {test_acc_lr[i]:.2f}%")
   

# ======================
# 任务4：卷积核可视化（第一层，显示8个以上）
# ======================
import matplotlib
matplotlib.use('Agg')  # 服务器环境不显示交互窗口
import matplotlib.pyplot as plt


# 用的是任务1训练好的模型
model.eval()  # 模型设为评估模式

# 获取第一层卷积核（conv1.weight的形状是 [out_channels, in_channels, kH, kW]）
conv1_weights = model.conv1.weight.data.cpu()  # shape: (32, 1, 3, 3)

# 可视化前16个卷积核（满足“至少8个”的要求）
plt.figure(figsize=(8, 4))
for i in range(16):
    plt.subplot(2, 8, i+1)
    # 取第i个卷积核，因为输入是单通道，所以squeeze去掉通道维度
    kernel = conv1_weights[i, 0, :, :]
    plt.imshow(kernel, cmap='gray')
    plt.axis('off')
plt.suptitle('First Layer Convolution Kernels (Conv1)', y=1.02)
plt.tight_layout()
plt.savefig("task4_conv1_kernels.png")
plt.close()

print("任务4完成：卷积核已保存为 task4_conv1_kernels.png")


# ======================
# 任务5：Feature Map可视化（第一层卷积输出）
# ======================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 选择一张测试图片（这里选第0张）
image, label = test_dataset[0]
image = image.unsqueeze(0).to(device)  # [1, 1, 28, 28]
print(f"选择的图片真实标签: {label}")

# 前向传播到第一层卷积，获取特征图
def get_feature_map(model, x, layer_name):
    feature_maps = []
    def hook(module, input, output):
        feature_maps.append(output.detach().cpu())
    
    # 注册钩子
    handle = getattr(model, layer_name).register_forward_hook(hook)
    model(x)
    handle.remove()
    return feature_maps[0]

# 获取conv1层的输出特征图
conv1_out = get_feature_map(model, image, 'conv1')  # shape: [1, 32, 28, 28]

# 可视化原图 + 前16个特征图（满足“至少8张”的要求）
plt.figure(figsize=(10, 5))

# 显示原图
plt.subplot(2, 8, 1)
plt.imshow(image.squeeze().cpu(), cmap='gray')
plt.title(f"Original (Label: {label})")
plt.axis('off')

# 显示特征图
for i in range(16):
    plt.subplot(2, 8, i+1)
    fm = conv1_out[0, i, :, :]
    plt.imshow(fm, cmap='gray')
    plt.title(f"FM {i+1}")
    plt.axis('off')

plt.suptitle('First Layer Feature Maps (Conv1)', y=1.02)
plt.tight_layout()
plt.savefig("task5_feature_maps.png")
plt.close()

print("任务5完成：特征图已保存为 task5_feature_maps.png")

# ======================
# 任务6：错误分类样本分析
# ======================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

model.eval()
wrong_images = []
wrong_labels = []
pred_labels = []

# 遍历测试集，找出所有预测错误的样本
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # 找出预测错误的样本
        mask = (predicted != labels)
        if mask.any():
            wrong_images.append(images[mask].cpu())
            wrong_labels.append(labels[mask].cpu())
            pred_labels.append(predicted[mask].cpu())

# 拼接所有错误样本
wrong_images = torch.cat(wrong_images, dim=0)
wrong_labels = torch.cat(wrong_labels, dim=0)
pred_labels = torch.cat(pred_labels, dim=0)

print(f"总共找到 {len(wrong_images)} 张错误分类图片")

# 显示前8张（满足题目“至少8张”的要求）
plt.figure(figsize=(12, 6))
n = min(8, len(wrong_images))
for i in range(n):
    plt.subplot(2, 4, i+1)
    plt.imshow(wrong_images[i].squeeze(), cmap='gray')
    plt.title(f"True: {wrong_labels[i]}\nPred: {pred_labels[i]}")
    plt.axis('off')

plt.suptitle('Misclassified Samples (True / Predicted)', y=1.02)
plt.tight_layout()
plt.savefig("task6_misclassified_samples.png")
plt.close()

print("任务6完成：错误样本图片已保存为 task6_misclassified_samples.png")

# ======================
# 任务7：混淆矩阵绘制
# ======================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

model.eval()
all_labels = []
all_preds = []

# 遍历测试集，收集所有真实标签和预测标签
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# 计算混淆矩阵
cm = confusion_matrix(all_labels, all_preds, labels=np.arange(10))

# 绘制混淆矩阵
plt.figure(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix on MNIST Test Set')
plt.tight_layout()
plt.savefig("task7_confusion_matrix.png")
plt.close()

print("任务7完成：混淆矩阵已保存为 task7_confusion_matrix.png")