import cv2
import mediapipe as mp
import numpy as np

# 初始化MediaPipe姿态检测
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def render_skeleton_video(input_video_path, output_save_path):
    # 打开输入视频
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {input_video_path}")
        return

    # 获取视频基础参数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_save_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # 颜色空间转换
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose_detector.process(rgb_frame)

        # 绘制全身33点骨架连线
        if result.pose_landmarks:
            mp_draw.draw_landmarks(
                image=frame,
                landmark_list=result.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=1)
            )
        writer.write(frame)

    # 释放资源
    cap.release()
    writer.release()
    print(f"✅ 骨架视频生成完成：{output_save_path}，总帧数{frame_count}")

if __name__ == "__main__":
    # 样本1：表现最优动作 forehand_lift 正手挑球
    vid1_in = "/home/ss/cv-course/bbb/forehand_lift/001.mp4"
    vid1_out = "/home/ss/cv-course/bbb/skeleton_lift.mp4"
    render_skeleton_video(vid1_in, vid1_out)

    # 样本2：表现最差动作 forehand_net_shot 正手网前小球
    vid2_in = "/home/ss/cv-course/bbb/forehand_net_shot/001.mp4"
    vid2_out = "/home/ss/cv-course/bbb/skeleton_net.mp4"
    render_skeleton_video(vid2_in, vid2_out)