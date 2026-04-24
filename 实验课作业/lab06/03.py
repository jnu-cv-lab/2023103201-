import cv2
import numpy as np

# ==============================================
# 任务1：ORB特征检测（已完成部分）
# ==============================================
img_box = cv2.imread('box.png', cv2.IMREAD_COLOR)
img_scene = cv2.imread('box_in_scene.png', cv2.IMREAD_COLOR)
gray_box = cv2.cvtColor(img_box, cv2.COLOR_BGR2GRAY)
gray_scene = cv2.cvtColor(img_scene, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures=1000)
kp_box, des_box = orb.detectAndCompute(gray_box, None)
kp_scene, des_scene = orb.detectAndCompute(gray_scene, None)

# ==============================================
# 任务2：ORB特征匹配（已完成部分）
# ==============================================
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_box, des_scene)
matches = sorted(matches, key=lambda x: x.distance)

# ==============================================
# 任务3：RANSAC剔除错误匹配
# ==============================================

# 1. 从匹配结果中提取两幅图像中的对应点坐标
src_pts = np.float32([kp_box[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# 2. 使用cv2.findHomography() + cv2.RANSAC方法
# 4. 设置重投影误差阈值为5.0
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 提取内点掩码，用于绘制匹配
matchesMask = mask.ravel().tolist()

# 5. 根据mask显示RANSAC后的内点匹配
draw_params = dict(
    matchColor=(0, 255, 0),         # 匹配线颜色：绿色
    singlePointColor=None,
    matchesMask=matchesMask,        # 只绘制内点
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
img_ransac = cv2.drawMatches(img_box, kp_box, img_scene, kp_scene, matches, None, **draw_params)

# 保存结果（无中文文件名）
cv2.imwrite('task3_ransac_matches.png', img_ransac)

# 6. 输出结果：内点数量、总匹配数量、内点比例
num_total = len(matches)
num_inliers = int(np.sum(mask))
inlier_ratio = num_inliers / num_total

print("===== Task 3 Result =====")
print("Homography Matrix:")
print(H)
print(f"Total matches: {num_total}")
print(f"RANSAC inliers: {num_inliers}")
print(f"Inlier ratio: {inlier_ratio:.4f}")

