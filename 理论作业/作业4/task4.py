import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

# 使用无界面后端
mpl.use('Agg')

plt.rcParams['axes.unicode_minus'] = False

# 计算图像梯度
def compute_gradient(img, operator='sobel'):
    if operator == 'sobel':
        # Sobel 水平、垂直梯度
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(sobel_x**2 + sobel_y**2)
    elif operator == 'prewitt':
        # Prewitt 水平、垂直梯度
        prewitt_x = cv2.filter2D(img, cv2.CV_64F, np.array([[-1,0,1],[-1,0,1],[-1,0,1]]))
        prewitt_y = cv2.filter2D(img, cv2.CV_64F, np.array([[-1,-1,-1],[0,0,0],[1,1,1]]))
        grad = np.sqrt(prewitt_x**2 + prewitt_y**2)
    elif operator == 'simple':
        # 简单中心差分梯度
        dx = np.zeros_like(img, dtype=np.float64)
        dy = np.zeros_like(img, dtype=np.float64)
        dx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2
        dy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
        grad = np.sqrt(dx**2 + dy**2)
    else:
        raise ValueError("operator must be 'sobel'/'prewitt'/'simple'")
    return grad.astype(np.float32)

# 梯度法计算 f_rms
def gradient_method_frms(block, grad_operator='sobel'):
    grad = compute_gradient(block, operator=grad_operator)
    E_grad2 = np.mean(grad**2)
    frms2 = E_grad2 / (4 * np.pi**2)
    return np.sqrt(frms2)

# FFT 法计算 95% 能量频率与 f_rms
def fft_method_95energy_frms(block, pixel_size=1.0):
    h, w = block.shape
    fft_block = np.fft.fft2(block)
    fft_shift = np.fft.fftshift(fft_block)
    power_spectrum = np.abs(fft_shift)**2

    # 构造频率网格
    fx = np.fft.fftfreq(w, d=pixel_size)
    fy = np.fft.fftfreq(h, d=pixel_size)
    fx = np.fft.fftshift(fx)
    fy = np.fft.fftshift(fy)
    FX, FY = np.meshgrid(fx, fy)
    F = np.sqrt(FX**2 + FY**2)

    # 计算总能量与累积95%能量对应频率
    total_energy = np.sum(power_spectrum)
    flat_F, flat_power = F.flatten(), power_spectrum.flatten()
    sort_idx = np.argsort(flat_F)
    sorted_F, sorted_power = flat_F[sort_idx], flat_power[sort_idx]
    cum_energy = np.cumsum(sorted_power)
    idx_95 = np.argmax(cum_energy >= 0.95 * total_energy)
    f_95 = sorted_F[idx_95]

    # 功率加权频率均方根
    f2_mean = np.sum(F**2 * power_spectrum) / total_energy
    frms_fft = np.sqrt(f2_mean)

    return f_95, frms_fft

# 将图像分块
def split_image_into_blocks(img, block_size=16):
    h, w = img.shape
    h_blocks = h // block_size
    w_blocks = w // block_size
    img_cropped = img[:h_blocks*block_size, :w_blocks*block_size]
    blocks = []
    for i in range(h_blocks):
        for j in range(w_blocks):
            block = img_cropped[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            blocks.append(block)
    return blocks, h_blocks, w_blocks

if __name__ == "__main__":
    img_path = "test.jpg"

    # 判断文件是否存在
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"File not found: {img_path}")

    # 灰度读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    print(f"Loaded image: {img_path}, shape: {img.shape}")

    block_size = 16
    blocks, h_blocks, w_blocks = split_image_into_blocks(img, block_size=block_size)
    print(f"Split into {h_blocks}x{w_blocks} blocks = {len(blocks)} blocks")

    grad_frms_list = []
    fft_95_list = []
    fft_frms_list = []

    # 逐块计算
    for idx, block in enumerate(blocks):
        # 梯度法
        frms_grad = gradient_method_frms(block, grad_operator='sobel')
        grad_frms_list.append(frms_grad)

        # FFT 法
        f_95, frms_fft = fft_method_95energy_frms(block, pixel_size=1.0)
        fft_95_list.append(f_95)
        fft_frms_list.append(frms_fft)

    grad_frms_arr = np.array(grad_frms_list)
    fft_95_arr = np.array(fft_95_list)
    fft_frms_arr = np.array(fft_frms_list)

    print("\n=== Results ===")
    print(f"Gradient f_rms:   mean = {np.mean(grad_frms_arr):.4f}, std = {np.std(grad_frms_arr):.4f}")
    print(f"FFT 95% energy:   mean = {np.mean(fft_95_arr):.4f}, std = {np.std(fft_95_arr):.4f}")
    print(f"FFT f_rms:        mean = {np.mean(fft_frms_arr):.4f}, std = {np.std(fft_frms_arr):.4f}")
    print(f"Corr(grad, fft95): {np.corrcoef(grad_frms_arr, fft_95_arr)[0,1]:.4f}")
    print(f"Corr(grad, fft_rms): {np.corrcoef(grad_frms_arr, fft_frms_arr)[0,1]:.4f}")

    # 散点对比图
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.scatter(grad_frms_arr, fft_95_arr, s=8, alpha=0.6)
    plt.xlabel("Gradient f_rms")
    plt.ylabel("FFT 95% energy frequency")
    plt.title("Gradient vs FFT 95% freq")
    plt.grid(True)
    m = max(grad_frms_arr.max(), fft_95_arr.max())
    plt.plot([0,m],[0,m], 'r--')

    plt.subplot(122)
    plt.scatter(grad_frms_arr, fft_frms_arr, s=8, alpha=0.6, c='orange')
    plt.xlabel("Gradient f_rms")
    plt.ylabel("FFT f_rms")
    plt.title("Gradient vs FFT f_rms")
    plt.grid(True)
    plt.plot([0,m],[0,m], 'r--')
    plt.tight_layout()
    plt.savefig("frequency_comparison.png", dpi=300)
    plt.close()

    # 频率分布直方图
    plt.figure(figsize=(12,4))
    plt.hist(grad_frms_arr, bins=30, alpha=0.5, label="Gradient f_rms", density=True)
    plt.hist(fft_95_arr, bins=30, alpha=0.5, label="FFT 95% freq", density=True)
    plt.hist(fft_frms_arr, bins=30, alpha=0.5, label="FFT f_rms", density=True)
    plt.xlabel("Frequency")
    plt.ylabel("Density")
    plt.title("Frequency distribution")
    plt.legend()
    plt.grid(True)
    plt.savefig("frequency_distribution.png", dpi=300)
    plt.close()

    # 热力图
    grad_map = grad_frms_arr.reshape(h_blocks, w_blocks)
    fft95_map = fft_95_arr.reshape(h_blocks, w_blocks)

    plt.figure(figsize=(12,5))
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title("Input image")
    plt.axis('off')

    plt.subplot(132)
    im1 = plt.imshow(grad_map, cmap='hot')
    plt.title("Gradient f_rms")
    plt.axis('off')
    plt.colorbar(im1, shrink=0.8)

    plt.subplot(133)
    im2 = plt.imshow(fft95_map, cmap='hot')
    plt.title("FFT 95% energy freq")
    plt.axis('off')
    plt.colorbar(im2, shrink=0.8)

    plt.tight_layout()
    plt.savefig("frequency_heatmap.png", dpi=300)
    plt.close()

    print("\nAll figures saved successfully.")