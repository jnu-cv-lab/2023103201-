import matplotlib
matplotlib.use('Agg')  # 强制无界面后端，专注保存图片
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def ycbcr_subsampling_reconstruction():
    # 1. 读取图像
    img_path = "test.jpg" 
    if not os.path.exists(img_path):
        print(f"❌ Error: Cannot find image {img_path}")
        return
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Error: Cannot read image {img_path}")
        return
    
    # 2. 色彩空间转换
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)

    # 3. 拆分通道
    Y = img_ycbcr[:, :, 0]
    Cb = img_ycbcr[:, :, 1]
    Cr = img_ycbcr[:, :, 2]

    # 4. 4:2:0 下采样 + 双线性插值重建
    height, width = Y.shape[:2]
    Cb_down = cv2.resize(Cb, (width//2, height//2), cv2.INTER_LINEAR)
    Cr_down = cv2.resize(Cr, (width//2, height//2), cv2.INTER_LINEAR)
    Cb_up = cv2.resize(Cb_down, (width, height), cv2.INTER_LINEAR)
    Cr_up = cv2.resize(Cr_down, (width, height), cv2.INTER_LINEAR)

    # 5. 重建图像 + 计算PSNR
    img_reconstructed_ycbcr = cv2.merge([Y, Cb_up, Cr_up])
    img_reconstructed = cv2.cvtColor(img_reconstructed_ycbcr, cv2.COLOR_YCrCb2RGB)
    psnr_value = cv2.PSNR(img_rgb.astype(np.uint8), img_reconstructed.astype(np.uint8))

    # 6. 绘制可视化图（全英文标题，避免方框）
    plt.figure(figsize=(18, 6))

    # 子图1：原始图像
    plt.subplot(1, 4, 1)
    plt.imshow(img_rgb)
    plt.title(f"Original\n{width}x{height}")  # 英文标题
    plt.axis('off')

    # 子图2：通道拆分
    plt.subplot(1, 4, 2)
    Cb_color = cv2.applyColorMap(cv2.normalize(Cb, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), cv2.COLORMAP_JET)
    Cr_color = cv2.applyColorMap(cv2.normalize(Cr, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), cv2.COLORMAP_HOT)
    Y_color = cv2.cvtColor(cv2.normalize(Y, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), cv2.COLOR_GRAY2BGR)
    channels_combined = np.hstack([Y_color, Cb_color, Cr_color])
    plt.imshow(channels_combined)
    plt.title("Y (Luma) | Cb (Blue) | Cr (Red)")  # 英文标题
    plt.axis('off')

    # 子图3：重建图像
    plt.subplot(1, 4, 3)
    plt.imshow(img_reconstructed)
    plt.title(f"Reconstructed\nPSNR: {psnr_value:.2f} dB")  # 英文标题
    plt.axis('off')

    # 子图4：差值图
    plt.subplot(1, 4, 4)
    diff = cv2.absdiff(img_rgb.astype(np.uint8), img_reconstructed.astype(np.uint8))
    plt.imshow(diff)
    plt.title("Difference Map")  # 英文标题
    plt.axis('off')

    # 保存图片
    save_path = "ycbcr_result.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 7. 打印结果
    print("=" * 60)
    print("✅ Job completed successfully!")
    print("=" * 60)
    print(f"📊 Results:")
    print(f"   - Image size: {width}x{height}")
    print(f"   - PSNR: {psnr_value:.2f} dB")
    print(f"💾 Image saved to: {os.getcwd()}/{save_path}")
    print("=" * 60)

if __name__ == "__main__":
    ycbcr_subsampling_reconstruction()