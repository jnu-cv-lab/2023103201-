import numpy as np
import torch
import matplotlib.pyplot as plt

# ====================== 1.【需求1】Sinusoidal Position Encoding 正弦位置编码 ======================
def sinusoidal_pe(seq_len: int, d_model: int):
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe

# ====================== 2.【需求2】二维向量旋转实现 ======================
def rotate_2d(x: np.ndarray, theta: float):
    rot_mat = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return rot_mat @ x

# ====================== 3.【需求3】高维RoPE实现 ======================
def precompute_rope_freq(seq_max: int, d_model: int):
    assert d_model % 2 == 0, "RoPE输入维度必须为偶数"
    theta = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pos = torch.arange(seq_max).unsqueeze(1)
    freqs = pos * theta
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor):
    B, L, D = x.shape
    x_complex = x.reshape(B, L, D // 2, 2).to(torch.float64)
    x_cis = torch.view_as_complex(x_complex)
    x_rot_cis = x_cis * freqs_cis[None, :, :]
    x_rot = torch.view_as_real(x_rot_cis).flatten(2)
    return x_rot.to(torch.float32)

# ====================== 5.【需求5】RoPE相对位置数值验证 ======================
def test_relative_rope():
    d = 64
    max_pos = 30
    freq = precompute_rope_freq(max_pos, d)
    q_base = torch.randn(1, d)
    k_base = torch.randn(1, d)

    def get_qk_dot(pos_q, pos_k):
        q_rot = apply_rope(q_base.unsqueeze(0), freq[pos_q:pos_q+1]) # shape [1,1,64]
        k_rot = apply_rope(k_base.unsqueeze(0), freq[pos_k:pos_k+1]) # shape [1,1,64]
        q_flat = q_rot.flatten()
        k_flat = k_rot.flatten()
        return torch.dot(q_flat, k_flat).item()

    dist = 3
    dot1 = get_qk_dot(5, 5 - dist)
    dot2 = get_qk_dot(12, 12 - dist)
    dot3 = get_qk_dot(20, 20 - dist)
    print(f"\n=====【需求5：RoPE相对位置验证】相对距离={dist} =====")
    print(f"Q=5,K=2 内积：{dot1:.4f}")
    print(f"Q=12,K=9 内积：{dot2:.4f}")
    print(f"Q=20,K=17 内积：{dot3:.4f}")
    print("结论：相对距离相同，QK内积几乎相等，验证RoPE相对位置特性\n")

# ====================== 主程序入口 ======================
if __name__ == "__main__":
    seq_len, d_model = 10, 64
    # 1. 测试正弦PE
    pe = sinusoidal_pe(seq_len, d_model)
    print("【需求1】正弦PE shape:", pe.shape)

    # 2. 测试二维旋转
    vec2 = np.array([1, 0])
    rot_vec = rotate_2d(vec2, np.pi/2)
    print("【需求2】2D向量[1,0]旋转90°结果：", np.round(rot_vec,3))

    # 3. 测试高维RoPE
    max_len = 20
    freq_cis = precompute_rope_freq(max_len, d_model)
    q = torch.randn(1, max_len, d_model)
    q_rope = apply_rope(q, freq_cis)
    print("【需求3】高维RoPE输出shape:", q_rope.shape)

    # 4.【需求4】E+pos 和 RoPE输入方式对比
    emb = torch.randn(1, seq_len, d_model)
    emb_add_pe = emb + pe.unsqueeze(0)
    print("\n【需求4】E+pos方式：emb直接+PE，输入阶段融合位置")
    wq, wk, wv = torch.randn(d_model,d_model), torch.randn(d_model,d_model), torch.randn(d_model,d_model)
    Q = emb @ wq
    K = emb @ wk
    V = emb @ wv
    Q_rot = apply_rope(Q, freq_cis[:seq_len])
    K_rot = apply_rope(K, freq_cis[:seq_len])
    print("RoPE方式：原始emb不变，Q/K旋转，V不做任何位置处理\n")

    test_relative_rope()

    # ==========【修复绘图索引】==========
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.imshow(pe.numpy(),cmap='coolwarm')
    plt.title("Sinusoidal PE")
    plt.xlabel("Dimension")
    plt.ylabel("Position")

    plt.subplot(122)
    v = torch.tensor([[1.,0.]])
    freq2 = precompute_rope_freq(5,2)
    for pos in range(5):
        p = apply_rope(v.unsqueeze(0),freq2[pos:pos+1])
        # 修正索引：p [B=1, L=1, D=2]
        plt.scatter(p[0,0,0], p[0,0,1], label=f'pos{pos}')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title("RoPE 2D Rotate")
    plt.tight_layout()
    plt.show()

    # 6.【需求6】RoPE优于E+pos文字说明
    print("=====【需求6】RoPE比E+pos更巧妙的原因 =====")
    print("1. 解耦内容与位置：E+pos直接相加导致语义和位置特征耦合，RoPE几何旋转实现内容、位置信息解耦；")
    print("2. 天然相对位置：E+pos仅编码绝对位置，RoPE的QK内积仅依赖相对偏移m-n，原生适配相对位置；")
    print("3. 超长外推更强：E+pos超出训练长度泛化差，RoPE依托三角函数周期性可外推未训练位置；")
    print("4. 灵活注入：E+pos仅在输入层一次性添加，RoPE仅作用QK，支持分层多头自定义位置编码。")