import numpy as np
import matplotlib.pyplot as plt

# ===================== 一维DFT与DCT-II的手动实现（纯Python，无第三方库） =====================
def dft_1d(x):
    """
    手动实现一维离散傅里叶变换 DFT
    输入：x 原始信号
    输出：DFT变换后的复数频谱
    """
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    # DFT公式：X[k] = 求和 x[n] * exp(-j*2πkn/N)
    e = np.exp(-2j * np.pi * k * n / N)
    return np.dot(e, x)

def idft_1d(X):
    """
    手动实现一维逆DFT
    输入：X DFT系数
    输出：重构后的原始信号
    """
    N = len(X)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(2j * np.pi * k * n / N)
    return np.dot(e, X) / N

def dct_ii_1d(x):
    """
    手动实现一维DCT-II变换（最常用的DCT形式）
    输入：x 原始信号
    输出：DCT实数系数
    """
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    
    # DCT归一化系数
    c = np.ones(N)
    c[0] = 1 / np.sqrt(2)
    
    # DCT基函数：余弦函数
    transform = np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    return np.sqrt(2 / N) * np.dot(transform * c, x)

def idct_ii_1d(X):
    """
    手动实现DCT-II逆变换
    输入：X DCT系数
    输出：重构信号
    """
    N = len(X)
    n = np.arange(N)
    k = n.reshape((N, 1))
    c = np.ones(N)
    c[0] = 1 / np.sqrt(2)
    transform = np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    return np.sqrt(2 / N) * np.dot(transform.T, X * c)

# ===================== 信号延拓方式 =====================
def dft_periodic_extension(x, repeat=2):
    """
    DFT隐含的周期延拓：直接重复原始信号
    特点：边界会出现跳变，产生高频分量
    """
    return np.tile(x, repeat)

def dct_even_symmetric_extension(x):
    """
    DCT隐含的偶对称延拓：镜像拼接信号
    特点：边界连续平滑，无高频跳变
    """
    return np.concatenate([x, x[::-1]])

# ===================== 能量计算 =====================
def calculate_energy_ratio(spectrum, num_coeffs):
    """
    计算前num_coeffs个系数的能量占总能量的比例
    用于对比能量集中性
    """
    energy_total = np.sum(np.abs(spectrum) ** 2)
    energy_part = np.sum(np.abs(spectrum[:num_coeffs]) ** 2)
    return energy_part / energy_total

# ===================== 主实验程序 =====================
if __name__ == "__main__":
    # 1. 生成测试信号：16点随机像素信号
    np.random.seed(42)
    x = np.random.randint(0, 256, size=16)
    N = len(x)
    print(f"原始信号长度：{N}")
    print(f"原始信号：{x}")

    # 2. 分别进行DFT周期延拓 和 DCT偶对称延拓
    x_dft_ext = dft_periodic_extension(x, repeat=2)
    x_dct_ext = dct_even_symmetric_extension(x)

    # 3. 计算DFT与DCT变换结果
    X_dft = dft_1d(x)
    X_dct = dct_ii_1d(x)

    # 4. 计算能量集中性：前k个系数的能量占比
    k_list = np.arange(1, N+1)
    energy_ratio_dft = [calculate_energy_ratio(X_dft, k) for k in k_list]
    energy_ratio_dct = [calculate_energy_ratio(X_dct, k) for k in k_list]

    # ===================== 绘图1：原始信号 + 两种延拓对比 =====================
    fig1, axs1 = plt.subplots(3, 1, figsize=(12, 10))
    
    axs1[0].stem(np.arange(N), x, basefmt=' ')
    axs1[0].set_title('Original Signal', fontsize=14)
    axs1[0].set_xlabel('Sample n', fontsize=12)
    axs1[0].set_ylabel('Amplitude', fontsize=12)
    axs1[0].grid(True, alpha=0.3)

    axs1[1].stem(np.arange(len(x_dft_ext)), x_dft_ext, basefmt=' ')
    axs1[1].axvline(x=N-0.5, color='r', linestyle='--', label='Extension Boundary')
    axs1[1].set_title('DFT Periodic Extension', fontsize=14)
    axs1[1].set_xlabel('Sample n', fontsize=12)
    axs1[1].set_ylabel('Amplitude', fontsize=12)
    axs1[1].legend()
    axs1[1].grid(True, alpha=0.3)

    axs1[2].stem(np.arange(len(x_dct_ext)), x_dct_ext, basefmt=' ')
    axs1[2].axvline(x=N-0.5, color='r', linestyle='--', label='Extension Boundary')
    axs1[2].set_title('DCT Even Symmetric Extension', fontsize=14)
    axs1[2].set_xlabel('Sample n', fontsize=12)
    axs1[2].set_ylabel('Amplitude', fontsize=12)
    axs1[2].legend()
    axs1[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('extension_comparison.png', dpi=300, bbox_inches='tight')

    # ===================== 绘图2：DFT与DCT频谱对比 =====================
    fig2, axs2 = plt.subplots(2, 1, figsize=(12, 8))
    
    axs2[0].stem(np.arange(N), np.abs(X_dft), basefmt=' ')
    axs2[0].set_title('DFT Spectrum (Magnitude)', fontsize=14)
    axs2[0].set_xlabel('Frequency k', fontsize=12)
    axs2[0].set_ylabel('|X[k]|', fontsize=12)
    axs2[0].grid(True, alpha=0.3)

    axs2[1].stem(np.arange(N), X_dct, basefmt=' ')
    axs2[1].set_title('DCT-II Spectrum', fontsize=14)
    axs2[1].set_xlabel('Frequency k', fontsize=12)
    axs2[1].set_ylabel('X[k]', fontsize=12)
    axs2[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('spectrum_comparison.png', dpi=300, bbox_inches='tight')

    # ===================== 绘图3：能量集中性对比 =====================
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(k_list, energy_ratio_dft, 'o-', label='DFT', linewidth=2, markersize=8)
    ax3.plot(k_list, energy_ratio_dct, 's-', label='DCT-II', linewidth=2, markersize=8)
    ax3.set_title('Energy Concentration Comparison', fontsize=16)
    ax3.set_xlabel('First k Coefficients', fontsize=14)
    ax3.set_ylabel('Energy Ratio', fontsize=14)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0.9, color='gray', linestyle='--', alpha=0.7, label='90% Energy Line')
    ax3.legend()

    plt.tight_layout()
    plt.savefig('energy_concentration.png', dpi=300, bbox_inches='tight')

    # ===================== 输出实验结果 =====================
    print("\n===== 能量集中性分析结果 =====")
    print(f"前4个系数能量占比：DFT = {energy_ratio_dft[3]:.2%}，DCT = {energy_ratio_dct[3]:.2%}")
    print(f"前8个系数能量占比：DFT = {energy_ratio_dft[7]:.2%}，DCT = {energy_ratio_dct[7]:.2%}")
    print(f"前12个系数能量占比：DFT = {energy_ratio_dft[11]:.2%}，DCT = {energy_ratio_dct[11]:.2%}")

    threshold = 0.9
    dft_k = next(k for k, r in enumerate(energy_ratio_dft) if r >= threshold) + 1
    dct_k = next(k for k, r in enumerate(energy_ratio_dct) if r >= threshold) + 1
    print(f"\n达到90%能量所需系数数量：DFT = {dft_k} 个，DCT = {dct_k} 个")

    plt.show()