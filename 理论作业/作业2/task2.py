import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===================== 自己实现的直方图均衡化 =====================
def my_histogram_equalization(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype(np.uint8)
    return cdf_normalized[img]

# ===================== 评价指标：PSNR + 信息熵 =====================
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def entropy(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    return -np.sum(hist * np.log2(hist + 1e-8))

# ===================== 一次性处理一张图的函数 =====================
def process_image(img_path, output_name):
    print(f"正在处理：{img_path}")
    img = cv2.imread(img_path, 0)  # 灰度读取

    # 各种增强方法
    eq_global = my_histogram_equalization(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)
    blur_mean = cv2.blur(img, (5,5))
    blur_gauss = cv2.GaussianBlur(img, (5,5), 1.0)
    blur_median = cv2.medianBlur(img, 5)
    
    # 锐化
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sharp = cv2.convertScaleAbs(laplacian) + img
    
    # 组合处理
    filter_eq = my_histogram_equalization(blur_gauss)
    eq_img = my_histogram_equalization(img)
    eq_filter = cv2.GaussianBlur(eq_img, (5,5), 1.0)

    # 所有方法列表
    methods = [
        ("Original", img),
        ("Global EQ", eq_global),
        ("CLAHE", clahe),
        ("Mean Blur", blur_mean),
        ("Gaussian Blur", blur_gauss),
        ("Median Blur", blur_median),
        ("Sharpen", sharp),
        ("Filter→EQ", filter_eq),
        ("EQ→Filter", eq_filter)
    ]

    # 绘制效果图
    plt.figure(figsize=(18, 12))
    for i, (name, res) in enumerate(methods, 1):
        plt.subplot(3, 3, i)
        plt.imshow(res, cmap="gray")
        p = psnr(img, res)
        e = entropy(res)
        plt.title(f"{name}\nPSNR:{p:.1f} | Entropy:{e:.1f}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{output_name}_result.png")
    plt.close()

    # 绘制直方图
    plt.figure(figsize=(18, 6))
    for i, (name, res) in enumerate(methods, 1):
        plt.subplot(3, 3, i)
        plt.hist(res.flatten(), 256, [0, 256])
        plt.title(f"{name} Hist")
    plt.tight_layout()
    plt.savefig(f"{output_name}_hist.png")
    plt.close()

# ===================== 一次性跑完三张图！！！=====================
process_image("img1.jpg", "img1")  # 低对比度图
process_image("img2.jpg", "img2")  # 噪声图
process_image("img3.jpg", "img3")  # 细节纹理图

print("✅ 全部处理完成！")