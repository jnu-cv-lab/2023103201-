import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 图像读入与预处理 =====================
img_path = "test.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"无法读取图像 {img_path}，请检查路径！")
h, w = img.shape
print(f"原始图像尺寸: {h} × {w}，灰度单通道")

# ===================== 2. 下采样（1/2 比例）=====================
scale = 2
new_h, new_w = h // scale, w // scale

img_down_naive = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
img_blur = cv2.GaussianBlur(img, (5, 5), sigmaX=1.5)
img_down_gauss = cv2.resize(img_blur, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

# ===================== 3. 图像恢复（上采样回原尺寸）=====================
img_up_nearest = cv2.resize(img_down_naive, (w, h), interpolation=cv2.INTER_NEAREST)
img_up_bilinear = cv2.resize(img_down_naive, (w, h), interpolation=cv2.INTER_LINEAR)
img_up_bicubic = cv2.resize(img_down_naive, (w, h), interpolation=cv2.INTER_CUBIC)
img_up_gauss_bilinear = cv2.resize(img_down_gauss, (w, h), interpolation=cv2.INTER_LINEAR)

# ===================== 4. 空间域质量评估：MSE & PSNR =====================
def calculate_mse_psnr(img1, img2, max_pixel=255.0):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        psnr = 100.0
    else:
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return mse, psnr

results = {}
results["Nearest Neighbor"] = calculate_mse_psnr(img, img_up_nearest)
results["Bilinear"] = calculate_mse_psnr(img, img_up_bilinear)
results["Bicubic"] = calculate_mse_psnr(img, img_up_bicubic)
results["Gaussian + Bilinear"] = calculate_mse_psnr(img, img_up_gauss_bilinear)

print("\n=== Spatial Domain Quality Evaluation (MSE/PSNR) ===")
for name, (mse, psnr) in results.items():
    print(f"{name}: MSE = {mse:.2f}, PSNR = {psnr:.2f} dB")

# ===================== 5. 傅里叶变换（DFT）分析 =====================
def dft_analysis(img):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    mag = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    mag_log = 20 * np.log(mag + 1e-8)
    mag_log = cv2.normalize(mag_log, None, 0, 255, cv2.NORM_MINMAX)
    return mag, mag_log

mag_ori, mag_log_ori = dft_analysis(img)
mag_down, mag_log_down = dft_analysis(img_down_naive)
mag_bi, mag_log_bi = dft_analysis(img_up_bilinear)

# ===================== 6. DCT 分析 =====================
def dct_analysis(img, low_freq_ratio=0.25):
    dct = cv2.dct(np.float32(img))
    dct_log = 20 * np.log(np.abs(dct) + 1e-8)
    dct_log = cv2.normalize(dct_log, None, 0, 255, cv2.NORM_MINMAX)
    
    h_dct, w_dct = dct.shape
    low_h, low_w = int(h_dct * low_freq_ratio), int(w_dct * low_freq_ratio)
    total_energy = np.sum(np.abs(dct) ** 2)
    low_energy = np.sum(np.abs(dct[:low_h, :low_w]) ** 2)
    energy_ratio = low_energy / total_energy
    
    return dct, dct_log, energy_ratio

low_freq_ratio = 0.25
_, dct_log_ori, ratio_ori = dct_analysis(img, low_freq_ratio)
_, dct_log_nn, ratio_nn = dct_analysis(img_up_nearest, low_freq_ratio)
_, dct_log_bi, ratio_bi = dct_analysis(img_up_bilinear, low_freq_ratio)
_, dct_log_bic, ratio_bic = dct_analysis(img_up_bicubic, low_freq_ratio)

print(f"\n=== DCT Low-Frequency Energy Ratio ({low_freq_ratio*100:.0f}% Region) ===")
print(f"Original: {ratio_ori:.2%}")
print(f"Nearest Neighbor: {ratio_nn:.2%}")
print(f"Bilinear: {ratio_bi:.2%}")
print(f"Bicubic: {ratio_bic:.2%}")

# ==========================================================
# ===================== 分图保存 ===========================
# ==========================================================

# ----------------- 图1：空间域图像 -----------------
plt.figure(figsize=(16, 8))
plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(img_down_naive, cmap='gray')
plt.title('Downsampled 1/2')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(img_up_nearest, cmap='gray')
plt.title('Nearest')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(img_up_bilinear, cmap='gray')
plt.title('Bilinear')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(img_up_bicubic, cmap='gray')
plt.title('Bicubic')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(img_up_gauss_bilinear, cmap='gray')
plt.title('Gauss+Bilinear')
plt.axis('off')

plt.tight_layout()
plt.savefig("01_spatial_domain.png", dpi=200, bbox_inches='tight')
plt.close()

# ----------------- 图2：DFT频谱 -----------------
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(mag_log_ori, cmap='gray')
plt.title('Original DFT')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mag_log_down, cmap='gray')
plt.title('Downsampled DFT')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(mag_log_bi, cmap='gray')
plt.title('Bilinear Restore DFT')
plt.axis('off')

plt.tight_layout()
plt.savefig("02_dft_spectrum.png", dpi=200, bbox_inches='tight')
plt.close()

# ----------------- 图3：DCT系数 -----------------
plt.figure(figsize=(16, 4))
plt.subplot(1, 4, 1)
plt.imshow(dct_log_ori, cmap='gray')
plt.title(f'Original DCT\n{ratio_ori:.2%}')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(dct_log_nn, cmap='gray')
plt.title(f'Nearest DCT\n{ratio_nn:.2%}')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(dct_log_bi, cmap='gray')
plt.title(f'Bilinear DCT\n{ratio_bi:.2%}')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(dct_log_bic, cmap='gray')
plt.title(f'Bicubic DCT\n{ratio_bic:.2%}')
plt.axis('off')

plt.tight_layout()
plt.savefig("03_dct_coeff.png", dpi=200, bbox_inches='tight')
plt.close()

# ----------------- 图4：指标对比 -----------------
plt.figure(figsize=(14, 8))

methods = list(results.keys())
mse_values = [results[m][0] for m in methods]
psnr_values = [results[m][1] for m in methods]
ratios = [ratio_ori, ratio_nn, ratio_bi, ratio_bic]
ratio_labels = ['Original', 'Nearest', 'Bilinear', 'Bicubic']

plt.subplot(2, 2, 1)
plt.bar(methods, mse_values, color=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
plt.title('MSE Comparison')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.subplot(2, 2, 2)
plt.bar(methods, psnr_values, color=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
plt.title('PSNR Comparison')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.subplot(2, 2, 3)
plt.bar(ratio_labels, ratios, color=['#66b3ff','#ff9999','#99ff99','#ffcc99'])
plt.title('DCT Low-Freq Energy Ratio')
plt.ylim(0, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("04_metrics.png", dpi=200, bbox_inches='tight')
plt.close()

# ===================== 结论 =====================
print("\n" + "="*80)
print("【Experimental Analysis Conclusion】")
print("1. Spatial Domain: Bicubic > Bilinear > Nearest Neighbor")
print("2. DFT: Downsampling causes aliasing (high frequency increase)")
print("3. DCT: Bicubic keeps low-frequency energy closest to original")
print("="*80)
