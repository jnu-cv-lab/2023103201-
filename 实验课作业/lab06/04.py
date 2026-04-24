import cv2
import numpy as np

# 任务1-3 前置代码
img_box = cv2.imread('box.png', cv2.IMREAD_COLOR)
img_scene = cv2.imread('box_in_scene.png', cv2.IMREAD_COLOR)
gray_box = cv2.cvtColor(img_box, cv2.COLOR_BGR2GRAY)
gray_scene = cv2.cvtColor(img_scene, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures=1000)
kp_box, des_box = orb.detectAndCompute(gray_box, None)
kp_scene, des_scene = orb.detectAndCompute(gray_scene, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_box, des_scene)
matches = sorted(matches, key=lambda x: x.distance)

src_pts = np.float32([kp_box[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# ===================== 任务4 代码（不显示窗口，只保存图片） =====================
h, w = img_box.shape[:2]
pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts, H)

# 画出目标框
img_result = cv2.polylines(img_scene.copy(), [np.int32(dst)], True, (0,0,255), 3)

# 只保存图片，不弹出窗口
cv2.imwrite('task4_target_localization.png', img_result)

print("Task4 completed, image saved as task4_target_localization.png")