import cv2
import numpy as np

# ==============================================
# 任务1 代码（你之前写的，保留在这里方便衔接）
# ==============================================
img_box = cv2.imread('box.png', cv2.IMREAD_COLOR)
img_scene = cv2.imread('box_in_scene.png', cv2.IMREAD_COLOR)
gray_box = cv2.cvtColor(img_box, cv2.COLOR_BGR2GRAY)
gray_scene = cv2.cvtColor(img_scene, cv2.COLOR_BGR2GRAY)

# 创建ORB检测器
orb = cv2.ORB_create(nfeatures=1000)
kp_box, des_box = orb.detectAndCompute(gray_box, None)
kp_scene, des_scene = orb.detectAndCompute(gray_scene, None)

# ==============================================
# 任务2：ORB特征匹配
# ==============================================

# 1. 使用cv2.BFMatcher()创建暴力匹配器
# 2. ORB描述子使用cv2.NORM_HAMMING
# 3. 使用crossCheck=True（双向匹配，过滤错误匹配）
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 进行匹配
matches = bf.match(des_box, des_scene)

# 4. 按照匹配距离从小到大排序
matches = sorted(matches, key=lambda x: x.distance)

# 6. 输出总匹配数量
print("===== Task 2 Result =====")
print(f"Total matches: {len(matches)}")

# 5. 显示前50个匹配结果（也可以改成30个）
num_show = 50  # 改为30就是前30个
img_match = cv2.drawMatches(
    img_box, kp_box, img_scene, kp_scene,
    matches[:num_show], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 保存匹配结果图（无中文，任务2专用）
cv2.imwrite('task2_orb_matches.png', img_match)

