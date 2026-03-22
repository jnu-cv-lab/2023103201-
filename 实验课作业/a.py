import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------- 任务1：使用 OpenCV 读取一张测试图片 ----------------------
img = cv2.imread("test.jpg")  # 请将 test.jpg 放在代码同目录下
if img is None:
    print("❌ 错误：无法读取图片，请检查文件路径是否正确！")
    exit()

# ---------------------- 任务2：输出图像基本信息 ----------------------
height, width, channels = img.shape
dtype = img.dtype
print("="*40)
print("📊 图像基本信息：")
print(f"  - 宽度：{width} 像素")
print(f"  - 高度：{height} 像素")
print(f"  - 通道数：{channels} 通道")
print(f"  - 数据类型：{dtype}")
print("="*40)

# ---------------------- 任务3：显示原图（Matplotlib 显示） ----------------------
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV 读取为 BGR，Matplotlib 需转为 RGB
plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')

# ---------------------- 任务4：转换为灰度图并显示 ----------------------
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(1, 2, 2)
plt.imshow(gray_img, cmap="gray")
plt.title("Grayscale Image")
plt.axis('off')
plt.tight_layout()
plt.savefig("result.png")  # 保存对比图
plt.show()

# ---------------------- 任务5：保存处理结果 ----------------------
cv2.imwrite("gray_test.jpg", gray_img)
print("✅ 灰度图已保存为：gray_test.jpg")

# ---------------------- 任务6：NumPy 简单操作（裁剪左上角区域） ----------------------
# 输出某个像素值（以 (100, 100) 为例）
pixel_value = img[100, 100]
print(f"📍 坐标 (100,100) 的像素值（BGR）：{pixel_value}")

# 裁剪左上角 100x100 区域
crop_img = img[0:100, 0:100]
cv2.imwrite("crop_test.jpg", crop_img)
print("✅ 左上角裁剪区域已保存为：crop_test.jpg")