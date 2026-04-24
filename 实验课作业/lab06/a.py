import cv2
import numpy as np
import time

# 读取图片
img_box = cv2.imread('box.png', cv2.IMREAD_COLOR)
img_scene = cv2.imread('box_in_scene.png', cv2.IMREAD_COLOR)
gray_box = cv2.cvtColor(img_box, cv2.COLOR_BGR2GRAY)
gray_scene = cv2.cvtColor(img_scene, cv2.COLOR_BGR2GRAY)

# ---------------------- ORB 实验（对比用） ----------------------
print("===== ORB 实验 =====")
start_time = time.time()

orb = cv2.ORB_create(nfeatures=1000)
kp1_orb, des1_orb = orb.detectAndCompute(gray_box, None)
kp2_orb, des2_orb = orb.detectAndCompute(gray_scene, None)

bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_orb = bf_orb.match(des1_orb, des2_orb)
matches_orb = sorted(matches_orb, key=lambda x: x.distance)

src_orb = np.float32([kp1_orb[m.queryIdx].pt for m in matches_orb]).reshape(-1,1,2)
dst_orb = np.float32([kp2_orb[m.trainIdx].pt for m in matches_orb]).reshape(-1,1,2)
H_orb, mask_orb = cv2.findHomography(src_orb, dst_orb, cv2.RANSAC, 5.0)

end_time = time.time()
time_orb = end_time - start_time

total_orb = len(matches_orb)
inliers_orb = int(np.sum(mask_orb))
ratio_orb = inliers_orb / total_orb if total_orb else 0

# 定位判断
h, w = img_box.shape[:2]
pts_box = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
pts_scene_orb = cv2.perspectiveTransform(pts_box, H_orb)
h_img, w_img = img_scene.shape[:2]
success_orb = True
for (x, y) in np.int32(pts_scene_orb).reshape(-1,2):
    if x < 0 or x >= w_img or y < 0 or y >= h_img:
        success_orb = False
if ratio_orb < 0.15:
    success_orb = False

img_orb = cv2.polylines(img_scene.copy(), [np.int32(pts_scene_orb)], True, (0,0,255), 3)
cv2.imwrite('task_orb_result.png', img_orb)

print(f"匹配数量：{total_orb}")
print(f"RANSAC内点数：{inliers_orb}")
print(f"内点比例：{ratio_orb:.4f}")
print(f"是否成功定位：{'是' if success_orb else '否'}")
print(f"运行时间：{time_orb:.4f}s")

# ---------------------- SIFT 实验（按题目要求） ----------------------
print("\n===== SIFT 实验 =====")
start_time = time.time()

# 1. 使用 cv2.SIFT_create()
sift = cv2.SIFT_create()
kp1_sift, des1_sift = sift.detectAndCompute(gray_box, None)
kp2_sift, des2_sift = sift.detectAndCompute(gray_scene, None)

# 2. 使用 cv2.NORM_L2 + 3. 使用 KNN matching
bf_sift = cv2.BFMatcher(cv2.NORM_L2)
matches = bf_sift.knnMatch(des1_sift, des2_sift, k=2)

# 4. 使用 Lowe ratio test 筛选匹配
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 5. 使用 RANSAC + Homography 定位
src_sift = np.float32([kp1_sift[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
dst_sift = np.float32([kp2_sift[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
H_sift, mask_sift = cv2.findHomography(src_sift, dst_sift, cv2.RANSAC, 5.0)

end_time = time.time()
time_sift = end_time - start_time

total_sift = len(good_matches)
inliers_sift = int(np.sum(mask_sift))
ratio_sift = inliers_sift / total_sift if total_sift else 0

# 定位判断
pts_scene_sift = cv2.perspectiveTransform(pts_box, H_sift)
success_sift = True
for (x, y) in np.int32(pts_scene_sift).reshape(-1,2):
    if x < 0 or x >= w_img or y < 0 or y >= h_img:
        success_sift = False
if ratio_sift < 0.15:
    success_sift = False

img_sift = cv2.polylines(img_scene.copy(), [np.int32(pts_scene_sift)], True, (0,0,255), 3)
cv2.imwrite('task_sift_result.png', img_sift)

print(f"匹配数量：{total_sift}")
print(f"RANSAC内点数：{inliers_sift}")
print(f"内点比例：{ratio_sift:.4f}")
print(f"是否成功定位：{'是' if success_sift else '否'}")
print(f"运行时间：{time_sift:.4f}s")

# ---------------------- 输出对比表格 ----------------------
print("\n===== ORB vs SIFT 对比表格 =====")
print("| 方法 | 匹配数量 | RANSAC内点数 | 内点比例 | 是否成功定位 | 运行时间(s) | 主观评价 |")
print("|------|----------|--------------|----------|--------------|-------------|----------|")
print(f"| ORB  | {total_orb:>8} | {inliers_orb:>12} | {ratio_orb:.4f} | {'是' if success_orb else '否':>12} | {time_orb:.4f} | 快，轻量 |")
print(f"| SIFT | {total_sift:>8} | {inliers_sift:>12} | {ratio_sift:.4f} | {'是' if success_sift else '否':>12} | {time_sift:.4f} | 慢，鲁棒 |")