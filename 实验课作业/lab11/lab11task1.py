import os
import cv2
import json
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ===================== 全局超参数 =====================
INPUT_DIM = 132
TARGET_FRAMES = 30
D_MODEL = 128
N_HEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
NUM_CLASSES = 6
DROPOUT = 0.1

BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 5
TEST_SIZE = 0.2

# 匹配bbb内六个子文件夹名称
LABEL_MAP = {
    0: "forehand_drive",
    1: "forehand_lift",
    2: "forehand_net_shot",
    3: "forehand_clear",
    4: "backhand_drive",
    5: "backhand_net_shot"
}
REV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"运行设备: {DEVICE}")

# MediaPipe 0.10.21 标准初始化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=0,
    smooth_landmarks=True
)

# -------------------------- 工具函数 --------------------------
def extract_pose_from_frame(frame):
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    landmarks = np.zeros((33, 4), dtype=np.float32)
    if results.pose_landmarks:
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            landmarks[idx] = [lm.x, lm.y, lm.z, lm.visibility]
    return landmarks.flatten()

def resample_frames(seq, target_len):
    ori_len = len(seq)
    if ori_len == target_len:
        return seq
    indices = np.linspace(0, ori_len - 1, target_len, dtype=int)
    return seq[indices]

def normalize_skeleton(seq):
    seq = seq.reshape(-1, 33, 4)
    hip_l = seq[:, 23, :2]
    hip_r = seq[:, 24, :2]
    hip_center = (hip_l + hip_r) / 2.0
    shoulder_l = seq[:, 11, :2]
    shoulder_r = seq[:, 12, :2]
    shoulder_width = np.linalg.norm(shoulder_l - shoulder_r, axis=-1, keepdims=True)
    shoulder_width[shoulder_width < 1e-6] = 1e-6
    seq[..., :2] -= hip_center[:, None, :]
    seq[..., :2] /= shoulder_width[:, None, :]
    return seq.reshape(-1, 132)

def process_video(video_path, target_frames=TARGET_FRAMES):
    cap = cv2.VideoCapture(video_path)
    frame_seq = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        feat = extract_pose_from_frame(frame)
        frame_seq.append(feat)
    cap.release()
    if len(frame_seq) == 0:
        return None
    frame_seq = np.array(frame_seq, dtype=np.float32)
    frame_seq = resample_frames(frame_seq, target_frames)
    frame_seq = normalize_skeleton(frame_seq)
    return frame_seq

# -------------------------- 带校验的数据集构建 --------------------------
def build_dataset(root_dir):
    all_data = []
    all_label = []
    print("根目录路径：", root_dir)
    print("目录存在：", os.path.exists(root_dir))
    for label_idx, cls_name in LABEL_MAP.items():
        cls_dir = os.path.join(root_dir, cls_name)
        print(f"\n文件夹 {cls_dir} 存在：{os.path.exists(cls_dir)}")
        if not os.path.exists(cls_dir):
            print(f"警告：缺失 {cls_name}")
            continue
        video_list = [f for f in os.listdir(cls_dir) if f.endswith((".mp4", ".avi"))]
        print(f"{cls_name} 视频数量：{len(video_list)}")
        for vid in tqdm(video_list, desc=cls_name):
            vid_path = os.path.join(cls_dir, vid)
            skeleton_seq = process_video(vid_path)
            if skeleton_seq is not None:
                all_data.append(skeleton_seq)
                all_label.append(label_idx)
            else:
                print(f"无效视频跳过：{vid}")
    print(f"\n有效样本总数：{len(all_data)}")
    if len(all_data) == 0:
        raise RuntimeError("无有效骨架数据，请检查路径与视频文件")
    X = np.array(all_data, dtype=np.float32)
    y = np.array(all_label, dtype=np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)
    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)
    with open("label_map.json", "w", encoding="utf-8") as f:
        json.dump(LABEL_MAP, f, ensure_ascii=False, indent=2)
    print("数据集文件保存完成")
    return X_train, X_test, y_train, y_test

# -------------------------- 数据集加载类 --------------------------
class BadmintonSkeletonDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path)
        self.labels = np.load(label_path)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        feat = torch.from_numpy(self.data[idx]).float()
        label = torch.tensor(self.labels[idx]).long()
        return feat, label

def get_dataloaders():
    train_set = BadmintonSkeletonDataset("X_train.npy", "y_train.npy")
    test_set = BadmintonSkeletonDataset("X_test.npy", "y_test.npy")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"训练批次:{len(train_loader)} | 测试批次:{len(test_loader)}")
    return train_loader, test_loader

# -------------------------- Transformer模型 --------------------------
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

class SkeletonTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, num_classes, dropout):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_classes))
    def forward(self, x):
        B, T, _ = x.shape
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        logits = self.classifier(x)
        return logits

model = SkeletonTransformer(
    input_dim=INPUT_DIM,
    d_model=D_MODEL,
    nhead=N_HEAD,
    num_layers=NUM_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -------------------------- 训练评估函数 --------------------------
def train_one_epoch(loader):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for feats, labels in tqdm(loader, desc="训练"):
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(feats)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * feats.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += feats.size(0)
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

def evaluate(loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for feats, labels in tqdm(loader, desc="测试"):
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            logits = model(feats)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += feats.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = total_correct / total_samples
    return acc, all_preds, all_labels

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_MAP.values(),
                yticklabels=LABEL_MAP.values())
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.title("混淆矩阵")
    plt.show()

# -------------------------- 单视频推理 --------------------------
def predict_single_video(video_path, model_path="badminton_transformer.pth"):
    infer_model = SkeletonTransformer(
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT
    ).to(DEVICE)
    infer_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    infer_model.eval()
    skeleton_seq = process_video(video_path)
    if skeleton_seq is None:
        print("视频提取失败")
        return
    input_tensor = torch.from_numpy(skeleton_seq).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = infer_model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
    pred_class = LABEL_MAP[pred_idx.item()]
    confidence = conf.item()
    print(f"预测动作：{pred_class} 置信度：{confidence:.2f}")
    return pred_class, confidence

# ===================== 程序入口 =====================
if __name__ == "__main__":
    print("=== 羽毛球动作识别 MediaPipe0.10.21+Transformer ===")
    # 你的bbb文件夹Linux绝对路径
    DATA_ROOT = "/home/ss/cv-course/bbb"

    # 1.生成数据集
    X_train, X_test, y_train, y_test = build_dataset(DATA_ROOT)

    # 2.训练模型
    train_loader, test_loader = get_dataloaders()
    for epoch in range(1, EPOCHS + 1):
        print(f"\n----- Epoch {epoch}/{EPOCHS} -----")
        train_loss, train_acc = train_one_epoch(train_loader)
        print(f"训练损失:{train_loss:.4f} 训练准确率:{train_acc:.4f}")

    # 3.测试评估
    test_acc, test_preds, test_labels = evaluate(test_loader)
    print(f"\n测试集准确率：{test_acc:.4f}")
    print("\n分类报告：")
    print(classification_report(test_labels, test_preds, target_names=LABEL_MAP.values()))
    plot_confusion_matrix(test_labels, test_preds)

    # 保存权重
    torch.save(model.state_dict(), "badminton_transformer.pth")
    print("模型权重已保存 badminton_transformer.pth")

    # 推理示例，自行填入视频路径
    # predict_single_video("/home/ss/cv-course/bbb/forehand_drive/test.mp4")