import cv2

# ---------------------- 任务1：ORB特征检测 ----------------------
# 1. 读取图像
# 读取目标图和场景图
img_box = cv2.imread('box.png', cv2.IMREAD_COLOR)
img_scene = cv2.imread('box_in_scene.png', cv2.IMREAD_COLOR)

# 转为灰度图（ORB算法通常在灰度图上运行）
gray_box = cv2.cvtColor(img_box, cv2.COLOR_BGR2GRAY)
gray_scene = cv2.cvtColor(img_scene, cv2.COLOR_BGR2GRAY)

# 2. 创建ORB检测器，设置nfeatures=1000
orb = cv2.ORB_create(nfeatures=1000)

# 3. 使用detectAndCompute()检测关键点和计算描述子
kp_box, des_box = orb.detectAndCompute(gray_box, None)
kp_scene, des_scene = orb.detectAndCompute(gray_scene, None)

# 4. 使用cv2.drawKeypoints()可视化关键点
# 在原图上绘制关键点（绿色）
img_kp_box = cv2.drawKeypoints(
    img_box, kp_box, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
img_kp_scene = cv2.drawKeypoints(
    img_scene, kp_scene, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# 保存可视化结果（无中文，任务编号清晰）
cv2.imwrite('task1_box_keypoints.png', img_kp_box)
cv2.imwrite('task1_scene_keypoints.png', img_kp_scene)

# 5. 输出两幅图像中的关键点数量
print("===== 任务1 结果 =====")
print(f"box.png 关键点数量：{len(kp_box)}")
print(f"box_in_scene.png 关键点数量：{len(kp_scene)}")

# 6. 输出描述子的维度
# 描述子维度：ORB默认是32（即32×8=256位二进制描述子）
print(f"描述子维度：{des_box.shape[1]}")

