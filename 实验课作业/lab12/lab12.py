import cv2
import numpy as np
import os

# 棋盘内角点规格、方格尺寸
corner_w = 9
corner_h = 6
square_size = 15.0
img_path = "./picture/"
out_path = "./calib_out/"
os.makedirs(out_path, exist_ok=True)

# 亚像素角点迭代终止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

obj_points = []
img_points = []
# 生成世界坐标系下棋盘格3D坐标
objp = np.zeros((corner_w * corner_h, 3), np.float32)
objp[:, :2] = np.mgrid[0:corner_w, 0:corner_h].T.reshape(-1, 2) * square_size

img_list = os.listdir(img_path)
valid_img_names = []
print(f"检测到图片总数：{len(img_list)}")

for fname in img_list:
    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        full_path = os.path.join(img_path, fname)
        img = cv2.imread(full_path)
        if img is None:
            print(f"跳过无法读取的文件: {fname}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 鲁棒角点检测配置
        ret, corners = cv2.findChessboardCorners(
            gray, (corner_w, corner_h),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH
                  + cv2.CALIB_CB_NORMALIZE_IMAGE
                  + cv2.CALIB_CB_FAST_CHECK
                  + cv2.CALIB_CB_FILTER_QUADS
        )

        if ret:
            corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners_sub)
            valid_img_names.append(fname)
            cv2.drawChessboardCorners(img, (corner_w, corner_h), corners_sub, ret)
            out_name = os.path.splitext(fname)[0] + "_corner.png"
            cv2.imwrite(os.path.join(out_path, out_name), img)
        else:
            print(f"{fname} 角点检测失败，舍弃该图")

if len(obj_points) == 0:
    raise Exception("未检测到任何有效棋盘图片，无法标定！")
print(f"\n有效参与标定图片数量：{len(obj_points)}")

# 执行相机标定
h, w = gray.shape[:2]
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

print("="*50)
print("相机标定结果：")
print(f"重投影总误差 RMS: {ret:.6f}")
print("\n相机内参矩阵 K：")
print(K)
print("\n畸变系数 dist [k1,k2,p1,p2,k3]：")
print(dist)

# 计算单张平均重投影误差
total_err = 0
for i in range(len(obj_points)):
    img_points_proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
    err = cv2.norm(img_points[i], img_points_proj, cv2.NORM_L2) / len(img_points_proj)
    total_err += err
avg_err = total_err / len(obj_points)
print(f"\n平均重投影误差: {avg_err:.6f} 像素")

# 选取第一张有效图做去畸变演示
test_img_name = valid_img_names[0]
test_img_path = os.path.join(img_path, test_img_name)
test_img = cv2.imread(test_img_path)
print(f"\n去畸变采用的原始图片：{test_img_name}")
h_test, w_test = test_img.shape[:2]
new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w_test, h_test), 1, (w_test, h_test))
map_x, map_y = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w_test, h_test), cv2.CV_32FC1)
img_undist = cv2.remap(test_img, map_x, map_y, cv2.INTER_LINEAR)
x, y, w_roi, h_roi = roi
img_crop = img_undist[y:y+h_roi, x:x+w_roi]

# 保存三组对比图
cv2.imwrite(os.path.join(out_path, "original_test.png"), test_img)
cv2.imwrite(os.path.join(out_path, "undistorted.png"), img_undist)
cv2.imwrite(os.path.join(out_path, "undist_cropped.png"), img_crop)
print(f"\n对比图片已保存至 {out_path}")

# 把参数写入文本方便粘贴报告
with open("camera_calib_result.txt", "w", encoding="utf-8") as f:
    f.write(f"RMS重投影误差: {ret:.6f}\n")
    f.write("内参矩阵 K:\n")
    f.write(str(K))
    f.write("\n畸变系数 dist:\n")
    f.write(str(dist))
    f.write(f"\n平均重投影误差: {avg_err:.6f}")
print("\n标定参数已保存至 camera_calib_result.txt")