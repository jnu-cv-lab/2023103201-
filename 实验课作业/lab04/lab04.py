import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ===================== 工具函数 =====================
def generate_checkerboard(size=256, grid=16):
    img = np.zeros((size, size), dtype=np.uint8)
    for i in range(0, size, grid):
        for j in range(0, size, grid):
            if (i//grid + j//grid) % 2 == 0:
                img[i:i+grid, j:j+grid] = 255
    return img

def generate_chirp(size=256):
    x = np.linspace(0, 8*np.pi, size)
    y = np.linspace(0, 8*np.pi, size)
    X, Y = np.meshgrid(x, y)
    img = np.sin(X + Y**2 / (2*size))
    img = ((img + 1) * 127.5).astype(np.uint8)
    return img

def compute_gradient(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(sobel_x**2 + sobel_y**2)
    return grad.astype(np.float32)

def compute_fft_spectrum(img):
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    mag = 20 * np.log(np.abs(fft_shift) + 1e-8)
    return mag

def downsample_direct(img, M=4):
    return img[::M, ::M]

def downsample_antialiased(img, M=4, sigma=1.8):
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    return blurred[::M, ::M]

# ===================== 第一部分：棋盘格 + Chirp 都生成 =====================
def part1_aliasing_test(M=4, sigma=1.8):
    checker = generate_checkerboard(256)
    chirp = generate_chirp(256)
    
    for name, img in [("Checkerboard", checker), ("Chirp", chirp)]:
        img_direct = downsample_direct(img, M)
        img_aa = downsample_antialiased(img, M, sigma)
        
        spec_ori = compute_fft_spectrum(img)
        spec_direct = compute_fft_spectrum(img_direct)
        spec_aa = compute_fft_spectrum(img_aa)
        
        plt.figure(figsize=(16,10))
        plt.subplot(2,3,1); plt.imshow(img, cmap='gray'); plt.title(f"Original {name}")
        plt.subplot(2,3,2); plt.imshow(img_direct, cmap='gray'); plt.title("Direct Downsample (Aliasing)")
        plt.subplot(2,3,3); plt.imshow(img_aa, cmap='gray'); plt.title("Gaussian + Downsample")
        plt.subplot(2,3,4); plt.imshow(spec_ori, cmap='jet'); plt.title("FFT Original")
        plt.subplot(2,3,5); plt.imshow(spec_direct, cmap='jet'); plt.title("FFT Aliasing")
        plt.subplot(2,3,6); plt.imshow(spec_aa, cmap='jet'); plt.title("FFT Anti-aliasing")
        plt.tight_layout()
        plt.savefig(f"part1_{name.lower()}.png", dpi=300)
        plt.close()
        print(f"✅ Part1 {name} 已保存")

# ===================== 第二部分：σ 验证 =====================
def part2_sigma_test(img, M=4):
    sigmas = [0.5, 1.0, 2.0, 4.0]
    plt.figure(figsize=(16,8))
    for i, s in enumerate(sigmas):
        blur = cv2.GaussianBlur(img, (0,0), sigmaX=s)
        down = blur[::M, ::M]
        plt.subplot(2,4,i+1); plt.imshow(blur, cmap='gray', interpolation='nearest'); plt.title(f"σ={s}")
        plt.subplot(2,4,i+5); plt.imshow(down, cmap='gray', interpolation='nearest'); plt.title(f"σ={s} down")
    plt.tight_layout()
    plt.savefig("part2_sigma.png", dpi=300)
    plt.close()
    print("✅ Part2 已保存")

# ===================== 第三部分：自适应下采样 =====================
def part3_adaptive(img, M=4, block=16):
    h, w = img.shape
    hb, wb = h//block, w//block
    crop = img[:hb*block, :wb*block]
    M_map = np.zeros((hb, wb))

    for i in range(hb):
        for j in range(wb):
            b = crop[i*block:(i+1)*block, j*block:(j+1)*block]
            g = np.mean(compute_gradient(b))
            localM = np.clip(M/(1+g/50), 1, M)
            M_map[i,j] = localM

    adapt_blur = np.zeros_like(crop)
    for i in range(hb):
        for j in range(wb):
            y1,y2 = i*block, (i+1)*block
            x1,x2 = j*block, (j+1)*block
            b = crop[y1:y2, x1:x2]
            s = 0.45 * M_map[i,j]
            adapt_blur[y1:y2, x1:x2] = cv2.GaussianBlur(b, (0,0), sigmaX=s)

    uni_blur = cv2.GaussianBlur(crop, (0,0), sigmaX=0.45*M)
    uni_down = uni_blur[::M, ::M]
    adapt_down = cv2.resize(adapt_blur, (wb, hb), interpolation=cv2.INTER_NEAREST)

    err_uni = np.abs(crop - cv2.resize(uni_down, crop.shape[::-1]))
    err_adp = np.abs(crop - cv2.resize(adapt_down, crop.shape[::-1]))

    plt.figure(figsize=(16,10))
    plt.subplot(231); plt.imshow(crop, cmap='gray', interpolation='nearest'); plt.title("Original")
    plt.subplot(232); plt.imshow(M_map, cmap='jet', interpolation='nearest'); plt.title("Local M Map")
    plt.subplot(233); plt.imshow(adapt_blur, cmap='gray', interpolation='nearest'); plt.title("Adaptive Blur")
    plt.subplot(234); plt.imshow(adapt_down, cmap='gray', interpolation='nearest'); plt.title("Adaptive Down")
    plt.subplot(235); plt.imshow(uni_down, cmap='gray', interpolation='nearest'); plt.title("Uniform Down")
    plt.subplot(236); plt.imshow(err_adp, cmap='hot', interpolation='nearest'); plt.title("Adaptive Error")
    plt.tight_layout()
    plt.savefig("part3_adaptive.png", dpi=300)
    plt.close()
    print("✅ Part3 已保存")

# ===================== 主程序：一次跑完 =====================
if __name__ == "__main__":
    print("===== 开始全部实验 =====")
    part1_aliasing_test(M=4, sigma=1.8)
    
    test_img = generate_chirp(256)
    part2_sigma_test(test_img, M=4)
    part3_adaptive(test_img, M=4)
    
    print("\n🎉 全部完成！")