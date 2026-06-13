import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 无GUI绘图后端
plt.switch_backend('Agg')
warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams["axes.unicode_minus"] = False

# ===================== 模型超参（优化调高正则） =====================
INPUT_DIM = 132
TARGET_FRAMES = 30
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
NUM_CLASSES = 6
DROPOUT = 0.2  # 原0.1加大防过拟合

# ===================== 训练超参（全套优化） =====================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3 
EPOCHS = 40  # 原20翻倍充分训练
WEIGHT_DECAY = 1e-4  # 新增L2正则
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"运行设备: {DEVICE}")

# 位置编码模块
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Transformer骨架模型（结构完全不变）
class SkeletonTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(INPUT_DIM, D_MODEL)
        self.pos_enc = PositionalEncoding(D_MODEL, dropout=DROPOUT)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, NUM_CLASSES)
        )
    def forward(self, x):
        B, T, _ = x.shape
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x_pool = x.mean(dim=0)
        logits = self.classifier(x_pool)
        return logits

# 数据集加载
class BadmintonDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path)
        self.labels = np.load(label_path)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        feat = torch.from_numpy(self.data[idx]).float()
        label = torch.tensor(self.labels[idx]).long()
        return feat, label

def get_data_loaders():
    train_dataset = BadmintonDataset("X_train.npy", "y_train.npy")
    test_dataset = BadmintonDataset("X_test.npy", "y_test.npy")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"训练批次数量: {len(train_loader)} | 测试批次数量: {len(test_loader)}")
    return train_loader, test_loader

# 单轮训练函数
def train_one_epoch(model, loader, loss_function, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for features, labels in tqdm(loader, desc="Training Progress"):
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        batch_logits = model(features)
        loss = loss_function(batch_logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * features.size(0)
        pred = torch.argmax(batch_logits, dim=1)
        total_correct += (pred == labels).sum().item()
        total_samples += features.size(0)
    avg_epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples
    return avg_epoch_loss, epoch_acc

# 测试评估
def test_model(model, loader, loss_function):
    model.eval()
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_true_labels = []
    with torch.no_grad():
        for features, labels in tqdm(loader, desc="Testing Progress"):
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            batch_logits = model(features)
            pred = torch.argmax(batch_logits, dim=1)
            total_correct += (pred == labels).sum().item()
            total_samples += features.size(0)
            all_predictions.extend(pred.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
    test_accuracy = total_correct / total_samples
    return test_accuracy, all_predictions, all_true_labels

# 保存混淆矩阵
def save_confusion_matrix(y_true, y_pred):
    action_names = [
        "forehand_drive",
        "forehand_lift",
        "forehand_net_shot",
        "forehand_clear",
        "backhand_drive",
        "backhand_net_shot"
    ]
    cm_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=action_names, yticklabels=action_names)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Confusion Matrix: Badminton Action Recognition")
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("混淆矩阵图片已保存: confusion_matrix.png")

# 绘制训练&测试损失、精度曲线（新增函数）
def draw_train_curve(epoch_list, train_loss_list, test_loss_list, train_acc_list, test_acc_list):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    # 损失曲线
    ax1.plot(epoch_list, train_loss_list, label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epoch_list, test_loss_list, label="Test Loss", color="#ff7f0e", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss Value")
    ax1.set_title("Loss Curve")
    ax1.legend()
    ax1.grid(alpha=0.3)
    # 精度曲线
    ax2.plot(epoch_list, train_acc_list, label="Train Accuracy", color="#2ca02c", linewidth=2)
    ax2.plot(epoch_list, test_acc_list, label="Test Accuracy", color="#d62728", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curve")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("train_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("训练曲线图片已保存: train_curve.png")

# ===================== 主程序（加入早停+学习率衰减+曲线记录） =====================
if __name__ == "__main__":
    print("===== 任务2 优化版模型训练程序 =====")
    model = SkeletonTransformer().to(DEVICE)
    loss_func = nn.CrossEntropyLoss()
    # 带L2权重衰减Adam
    optim = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # 学习率衰减：每15轮乘以0.7
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=15, gamma=0.7)
    train_loader, test_loader = get_data_loaders()

    best_acc = 0.0
    best_weight_path = "badminton_transformer_best.pth"
    patience = 8
    patience_count = 0

    # 初始化列表，记录每一轮指标，用于绘制曲线
    epoch_record = []
    train_loss_record = []
    test_loss_record = []
    train_acc_record = []
    test_acc_record = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\n----- Epoch {epoch}/{EPOCHS} -----")
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_func, optim)
        # 每轮跑完立刻测测试集精度
        test_acc, _, _ = test_model(model, test_loader, loss_func)
        # 计算测试集平均loss用于曲线绘制
        test_total_loss = 0.0
        model.eval()
        with torch.no_grad():
            for feat, lab in test_loader:
                feat, lab = feat.to(DEVICE), lab.to(DEVICE)
                log = model(feat)
                test_total_loss += loss_func(log, lab).item() * feat.shape[0]
        test_avg_loss = test_total_loss / len(test_loader.dataset)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
        # 存入记录列表
        epoch_record.append(epoch)
        train_loss_record.append(train_loss)
        test_loss_record.append(test_avg_loss)
        train_acc_record.append(train_acc)
        test_acc_record.append(test_acc)

        # 保存最优权重
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_weight_path)
            print(f"✅ 新最优精度 {best_acc:.4f}，已保存权重")
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"⏹ 连续{patience}轮无提升，提前终止训练")
                break
        scheduler.step()

    # 训练结束绘制曲线
    draw_train_curve(epoch_record, train_loss_record, test_loss_record, train_acc_record, test_acc_record)

    # 加载全局最优权重做最终评估
    model.load_state_dict(torch.load(best_weight_path, map_location=DEVICE))
    final_acc, pred_list, true_list = test_model(model, test_loader, loss_func)

    print(f"\n===== 最终最优测试结果 =====")
    print(f"Test Set Accuracy: {final_acc:.4f}")

    action_names = [
        "forehand_drive",
        "forehand_lift",
        "forehand_net_shot",
        "forehand_clear",
        "backhand_drive",
        "backhand_net_shot"
    ]
    print("\nClassification Report:")
    print(classification_report(true_list, pred_list, target_names=action_names, zero_division=0))

    save_confusion_matrix(true_list, pred_list)
    # 同时复制一份常规命名权重给推理脚本用
    torch.save(model.state_dict(), "badminton_transformer.pth")
    print(f"\n训练完成！最优精度{best_acc:.4f}，权重已保存为 badminton_transformer.pth")