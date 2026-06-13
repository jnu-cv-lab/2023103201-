import os
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn

# ===================== 全局超参 和任务2模型完全统一 =====================
INPUT_DIM = 132
TARGET_FRAMES = 30
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
NUM_CLASSES = 6
DROPOUT = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 动作标签映射
LABEL_DICT = {
    0: "forehand_drive",
    1: "forehand_lift",
    2: "forehand_net_shot",
    3: "forehand_clear",
    4: "backhand_drive",
    5: "backhand_net_shot"
}

# -------------------------- 位置编码（和训练模型结构完全一致） --------------------------
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

# -------------------------- 复刻任务2的Transformer模型结构 --------------------------
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

# -------------------------- MediaPipe初始化 提取人体姿态骨架 --------------------------
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=True
)

# 1. 单帧提取132维骨架特征
def extract_single_frame_skeleton(frame):
    h, w, _ = frame.shape
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose_model.process(rgb_img)
    skeleton = np.zeros((33, 4), dtype=np.float32)
    if result.pose_landmarks:
        for idx, landmark in enumerate(result.pose_landmarks.landmark):
            skeleton[idx][0] = landmark.x
            skeleton[idx][1] = landmark.y
            skeleton[idx][2] = landmark.z
            skeleton[idx][3] = landmark.visibility
    return skeleton.flatten()

# 2. 时序重采样统一到30帧
def resample_sequence(seq, target_len):
    ori_len = len(seq)
    if ori_len == target_len:
        return seq
    sample_idx = np.linspace(0, ori_len - 1, target_len, dtype=int)
    return seq[sample_idx]

# 3. 骨架归一化（和预处理逻辑完全一致：髋中点为原点，肩宽缩放）
def normalize_skeleton(seq_arr):
    seq_arr = seq_arr.reshape(-1, 33, 4)
    # 左右髋关键点
    hip_left = seq_arr[:, 23, :2]
    hip_right = seq_arr[:, 24, :2]
    hip_center = (hip_left + hip_right) / 2.0
    # 左右肩计算缩放尺度
    shoulder_left = seq_arr[:, 11, :2]
    shoulder_right = seq_arr[:, 12, :2]
    shoulder_width = np.linalg.norm(shoulder_left - shoulder_right, axis=1, keepdims=True)
    shoulder_width[shoulder_width < 1e-6] = 1e-6
    # 平移+缩放
    seq_arr[..., :2] -= hip_center[:, None, :]
    seq_arr[..., :2] /= shoulder_width[:, None, :]
    return seq_arr.reshape(-1, 132)

# -------------------------- 完整视频转30×132骨架序列 --------------------------
def video_to_skeleton_sequence(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_feature_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        feat_132 = extract_single_frame_skeleton(frame)
        frame_feature_list.append(feat_132)
    cap.release()
    if len(frame_feature_list) == 0:
        raise RuntimeError("视频读取失败，无有效帧")
    seq_np = np.array(frame_feature_list, dtype=np.float32)
    seq_np = resample_sequence(seq_np, TARGET_FRAMES)
    seq_np = normalize_skeleton(seq_np)
    return seq_np

# -------------------------- 单样本推理主函数 --------------------------
def run_inference(demo_video_path, weight_file="badminton_transformer.pth"):
    # 加载训练好的模型权重
    model = SkeletonTransformer().to(DEVICE)
    model.load_state_dict(torch.load(weight_file, map_location=DEVICE))
    model.eval()

    # 流程1：demo视频 → 骨架序列 [30,132]
    skeleton_sample = video_to_skeleton_sequence(demo_video_path)
    # 扩充batch维度 [1,30,132]
    input_tensor = torch.from_numpy(skeleton_sample).float().unsqueeze(0).to(DEVICE)

    # 流程2：模型前向推理、softmax求概率
    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.softmax(logits, dim=1)
        max_conf, pred_index = torch.max(prob, dim=1)

    pred_class_name = LABEL_DICT[pred_index.item()]
    confidence_value = max_conf.item()
    # 按实验示例格式打印输出
    print(f"Predicted class: {pred_class_name}")
    print(f"Confidence: {confidence_value:.2f}")
    return pred_class_name, confidence_value

# ===================== 运行入口 =====================
if __name__ == "__main__":
    print("===== 任务3：单视频推理程序 =====")
    # 修改为你的demo视频绝对路径
    DEMO_VIDEO_PATH = "/home/ss/cv-course/bbb/forehand_clear/004.mp4"
    run_inference(DEMO_VIDEO_PATH)