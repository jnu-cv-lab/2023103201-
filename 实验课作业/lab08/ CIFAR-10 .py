# ==============================
# CIFAR-10 完整实验代码（任务1~进阶3）
# ==============================
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# ======================
# 设备配置
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# 生成 CIFAR-10 风格模拟数据（彩色3通道）
# ======================
# 训练集
train_images = torch.randn(50000, 3, 32, 32)
train_labels = torch.randint(0, 10, (50000,))
for i in range(10):
    mask = train_labels == i
    train_images[mask] += i * 0.2

# 验证集
val_images = torch.randn(10000, 3, 32, 32)
val_labels = torch.randint(0, 10, (10000,))
for i in range(10):
    mask = val_labels == i
    val_images[mask] += i * 0.2

# 测试集
test_images = torch.randn(10000, 3, 32, 32)
test_labels = torch.randint(0, 10, (10000,))
for i in range(10):
    mask = test_labels == i
    test_images[mask] += i * 0.2

# 加载器
train_loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(val_images, val_labels), batch_size=64, shuffle=False)
test_loader = DataLoader(TensorDataset(test_images, test_labels), batch_size=8, shuffle=False)

# ======================
# CIFAR-10 专用 CNN 模型（适配彩色图）
# ======================
class CIFAR_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
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

model = CIFAR_CNN().to(device)

# ======================
# 训练设置
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5

train_loss = []
train_acc = []
val_loss = []
val_acc = []

# ======================
# 训练 + 验证循环
# ======================
for epoch in range(epochs):
    # 训练
    model.train()
    tl, ta = 0,0
    for img, lab in train_loader:
        img, lab = img.to(device), lab.to(device)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, lab)
        loss.backward()
        optimizer.step()
        tl += loss.item()
        ta += (out.argmax(1)==lab).sum().item()
    tl /= len(train_loader)
    ta = 100*ta/50000
    train_loss.append(tl)
    train_acc.append(ta)

    # 验证
    model.eval()
    vl, va = 0,0
    with torch.no_grad():
        for img, lab in val_loader:
            img, lab = img.to(device), lab.to(device)
            out = model(img)
            vl += criterion(out, lab).item()
            va += (out.argmax(1)==lab).sum().item()
    vl /= len(val_loader)
    va = 100*va/10000
    val_loss.append(vl)
    val_acc.append(va)

    print(f"Epoch {epoch+1} | Train Acc {ta:.2f}% | Val Acc {va:.2f}%")

# ======================
# 测试集
# ======================
model.eval()
test_correct = 0
with torch.no_grad():
    for img, lab in test_loader:
        img, lab = img.to(device), lab.to(device)
        test_correct += (model(img).argmax(1)==lab).sum().item()
test_acc = 100*test_correct/10000
print("\n测试集准确率:", round(test_acc,2),"%")

# ======================
# 画图（任务7）
# ======================
plt.figure(figsize=(12,5))
plt.subplot(121)
plt.plot(train_loss, label="train loss")
plt.plot(val_loss, label="val loss")
plt.title("Loss")
plt.legend()

plt.subplot(122)
plt.plot(train_acc, label="train acc")
plt.plot(val_acc, label="val acc")
plt.title("Accuracy")
plt.legend()
plt.savefig("cifar10_train_val.png")
plt.show()

# ======================
# 输出8张测试图（任务6）
# ======================
imgs, labs = next(iter(test_loader))
preds = model(imgs.to(device)).argmax(1)

plt.figure(figsize=(12,5))
for i in range(8):
    plt.subplot(2,4,i+1)
    img = imgs[i].permute(1,2,0)
    img = (img - img.min())/(img.max()-img.min())
    plt.imshow(img)
    plt.title(f"True:{labs[i]}\nPred:{preds[i]}")
    plt.axis("off")
plt.savefig("cifar10_test.png")
plt.show()