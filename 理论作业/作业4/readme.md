
# 作业：图像局部频率分析（梯度法 vs FFT法）

## 作业任务要求
> 对图像分块，对每个块做 FFT，找到包含 95% 能量的最高频率：对比用梯度近似得到的最高频率，看一看一致性如何？

---

## 空域梯度方法的完整流程是：

$$
I(x,y) \xrightarrow{\text{空域差分}} |\nabla I| \xrightarrow{\text{统计}} E[||\nabla I|^2] \xrightarrow{:4\pi^2 Var(I)} f_{rms}^2
$$

全程在空域，不碰 FFT。差分算子的选择就决定了精度。

---

## FFT 方法：

$$
I(x,y) \xrightarrow{\text{FFT}} F[k] \xrightarrow{\text{功率谱}} P[k] \xrightarrow{\text{二阶矩}} f_{rms}^2
$$

全程在频域。

---

## 理论说明
图像局部方差/梯度描述的是内容的变化程度，$\sigma_{kernel}$ 描述的是滤波器的宽度。两者通过“梯度 → 局部最高频率 → M → $\sigma_{kernel}$”这条链连接起来。
梯度只是局部频率的一个近似估计，真正严格的做法是直接用局部 FFT 测量实际频率，再反推所需的滤波器参数。
