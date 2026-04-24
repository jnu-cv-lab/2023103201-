import cv2
import numpy as np

# 读取图片
img_box = cv2.imread('box.png', cv2.IMREAD_COLOR)
img_scene = cv2.imread('box_in_scene.png', cv2.IMREAD_COLOR)
gray_box = cv2.cvtColor(img_box, cv2.COLOR_BGR2GRAY)
gray_scene = cv2.cvtColor(img_scene, cv2.COLOR_BGR2GRAY)

# 要测试的参数
nfeatures_list = [500, 1000, 2000]
results = []

for n in nfeatures_list:
    print(f"\n===== nfeatures = {n} =====")

    # 1. ORB 特征检测
    orb = cv2.ORB_create(nfeatures=n)
    kp1, des1 = orb.detectAndCompute(gray_box, None)
    kp2, des2 = orb.detectAndCompute(gray_scene, None)

    # 2. 匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 3. RANSAC
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    total = len(matches)
    inliers = int(np.sum(mask))
    ratio = inliers / total if total != 0 else 0

    # 4. 定位
    h, w = img_box.shape[:2]
    pts_box = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    pts_scene = cv2.perspectiveTransform(pts_box, H)

    # ===================== 正确判断：是否成功定位 =====================
    h_img, w_img = img_scene.shape[:2]
    success = True
    for (x, y) in np.int32(pts_scene).reshape(-1, 2):
        if x < 0 or x >= w_img or y < 0 or y >= h_img:
            success = False  # 点跑出图片 = 定位失败
    if ratio < 0.15:
        success = False      # 内点比例太低 = 定位失败
    # =================================================================

    # 画图保存（不弹窗口）
    img_res = cv2.polylines(img_scene.copy(), [np.int32(pts_scene)], True, (0,0,255), 3)
    cv2.imwrite(f'task6_result_{n}.png', img_res)

    # 输出
    print(f"模板关键点：{len(kp1)}")
    print(f"场景关键点：{len(kp2)}")
    print(f"总匹配：{total}")
    print(f"内点：{inliers}")
    print(f"内点比例：{ratio:.4f}")
    print(f"成功定位：{'是' if success else '否'}")

    results.append((n, len(kp1), len(kp2), total, inliers, round(ratio,4), '是' if success else '否'))

# 打印最终表格（直接复制进报告）
print("\n\n===== 实验结果表格 =====")
print("| nfeatures | 模板关键点 | 场景关键点 | 总匹配 | 内点 | 内点比例 | 成功定位 |")
print("|-----------|------------|------------|--------|------|----------|----------|")
for r in results:
    print(f"| {r[0]:>9} | {r[1]:>10} | {r[2]:>10} | {r[3]:>6} | {r[4]:>4} | {r[5]:>8} | {r[6]:>8} |")