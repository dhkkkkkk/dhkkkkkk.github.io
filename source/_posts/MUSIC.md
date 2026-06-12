---
title: 声源定位之MUSIC
date: 2026-06-05 15:32:23
tags: 麦克风阵列算法
mathjax: true
---

# 协方差矩阵

协方差矩阵描述了各个麦克风接收信号之间的相关性和空间结构：
$$
\mathbf{R}_{xx} = E[\mathbf{X}(t)\mathbf{X}^H(t)]=\begin{bmatrix} E[x_1 x_1^*] & E[x_1 x_2^*] & \dots \\ E[x_2 x_1^*] & E[x_2 x_2^*] & \dots \\ \vdots & \vdots & \ddots \end{bmatrix}
$$

* E：两个麦克风的数学期望，通过<u>多快拍取均值替代</u>
* R是Hermitian矩阵，不是对称矩阵
* 对角线：表示了该麦克风接收到的平均能量
* 其他元素：两个麦克风接收信号的相关程度，数值越大越相关（分正负，不相关则接近0）

# MUSIC（Multiple Signal Classification，多信号分类）

MUSIC**算法是空间谱估计领域的里程碑式技术。与 GCC-PHAT 或 SRP-PHAT 等基于时延估计（TDOA）的波束形成技术不同，MUSIC 是一种基于**特征空间分解（Subspace Decomposition）的高分辨率 DOA 估计算法。

对于协方差矩阵，可以描述为：
$$
\mathbf{R}_{xx} = E[\mathbf{X}(t)\mathbf{X}^H(t)] = \mathbf{A}(\theta) \mathbf{R}_{ss} \mathbf{A}^H(\theta) + \sigma^2 \mathbf{I}_M
$$
σ ^2是噪声方差，I 是单位矩阵

## 特征值分解

MUSIC的核心就是对协方差矩阵进行特征值分解：
$$
R=UΛU^H
$$
对于特征值分解的理解，当一个特征值和特征向量表示为：
$$
Ru_i=\lambda u_i
$$
则代表当R作用于方向u时，不会改变方向，只会放大λ倍，因此在阵列信号中：

* 特征向量表示方向
* 特征值则表示该方向上的能量大小

所以大特征值对应方向则为声源方向

由于协方差矩阵为Hermitian矩阵，因此对于MxM的R，必定有M个特征值，**<u>且所有特征向量彼此正交</u>**

当得到M个特征值后，**根据大小排序**，将对应的特征向量划分为信号子空间和噪声子空间。（这个划分规则需要人为指定声源个数K）
$$
\mathbf{U}_s = [\mathbf{u}_1, \mathbf{u}_2, \dots, \mathbf{u}_K]
$$

$$
\mathbf{U}_n = [\mathbf{u}_{K+1}, \mathbf{u}_{K+2}, \dots, \mathbf{u}_M]
$$

理想情况下，噪声特征值等于方差σ ^2

## 空间谱函数

刚刚我们提到了两个重点：

* 特征向量表示方向信息
* 所有特征向量彼此正交

因此，对于提取到的噪声子空间中的特征向量，理想情况存在：
$$
\mathbf{a}^H(\theta_k)\mathbf{u}_i = 0,  i = K+1, \dots, M
$$
a为声源方向导向矢量。

这意味着，真实声源方向的导向向量在噪声子空间上的投影模长为 0：
$$
\mathbf{a}^H(\theta_k) \mathbf{U}_n \mathbf{U}_n^H \mathbf{a}(\theta_k) = 0
$$
在真实环境中，我们则希望该值最小，因此我们定义一个MUSIC空间谱函数：
$$
P_{\text{MUSIC}}(\theta) = \frac{1}{\mathbf{a}^H(\theta) \mathbf{U}_n \mathbf{U}_n^H \mathbf{a}(\theta)}
$$
通过对θ进行扫描，当空间谱出现极大值时，则说明扫描到了真实声源

# 代码实现

```python
class MUSIC():
    def __init__(self,
                 mic_struct,
                 data,
                 num_freqs,
                 sampling_frequency=16000,
                 sound_speed=343):
        self.num_freqs = num_freqs
        self.data = data
        self.sampling_frequency = sampling_frequency
        self.freqbin = np.fft.rfftfreq(self.num_freqs, d=1.0/self.sampling_frequency)
        self.angle_hop = 1.0

        self.sound_speed = sound_speed
        self.mic_struct = mic_struct
        self.Nchan = len(self.mic_struct)

        
	#各方向的阵列绝对时延（参考为0坐标点）
    def get_tdoa(self, phi, theta):
        # phi=0为+x，=180为-x
        # theta=0为+z =180为-z

        a = np.array([  np.sin(theta) * np.cos(phi) ,
                        np.sin(theta) * np.sin(phi),
                        np.cos(theta)])

        tdoa = -np.sum(self.mic_struct * a[None, :], keepdims=True, axis=1) / self.sound_speed
        return tdoa
    
    
        def my_music(self,
                    theta_range=None,
                    phi_range=None,
                    freq_range=None,
                    n_sources=2
                    ):

        if freq_range is None:
            freq_range = [1000, 2000]
        if phi_range is None:
            phi_range = [30, 150]
        if theta_range is None:
            theta_range = [30, 150]


        mask = (self.freqbin >= freq_range[0]) & (self.freqbin <= freq_range[1])
        data_valid = self.data[:,:,mask]
        freqs = self.freqbin[mask]

        theta_angles = np.arange(theta_range[0], theta_range[1] + 1, self.angle_hop)
        phi_angles = np.arange(phi_range[0], phi_range[1] + 1, self.angle_hop)
        thetaRads = np.deg2rad(theta_angles)
        phiRads = np.deg2rad(phi_angles)


        doa_matrix = np.zeros((self.Nchan, len(thetaRads), len(phiRads)))
        for j, theta in enumerate(thetaRads):
            for i, phi in enumerate(phiRads):
                doa_matrix[:, j, i] = self.get_tdoa(phi, theta)[:, 0]

        #展平计算更快        
        doa_flat = doa_matrix.reshape(self.Nchan,-1)
        music_spectrum_flat = np.zeros(len(theta_angles) *len(phi_angles))
        
        #每个频率分开计算后求和
        for f_idx, freq in enumerate(tqdm(freqs, desc="MUSICing")):
            X_f = data_valid[:,:, f_idx]    
            cross = X_f.conj().T @ X_f #协方差矩阵
            eig_vals, eig_vecs = np.linalg.eigh(cross)	#特征分解
            Un = eig_vecs[:,:-n_sources]	#噪声子空间
            a = np.exp(-1j * 2 * np.pi * freq * doa_flat) #导向矢量，因为tdoa返回的是时延值，所以需要取负
            denominator = np.sum(np.abs(Un.conj().T @ a) ** 2, axis=0) #导向矢量在噪声空间的投影模长（多快拍在此求和）
            denominator = np.maximum(denominator, 1e-18) #防止分母为0
            music_spectrum_flat += 1.0 / denominator

        music_map = music_spectrum_flat.reshape((len(thetaRads), len(phiRads)))
```

{% asset_img music.png This is an image %} 

可以和上一章中的SRP-phat和DAS比较，可以发现：

* 由于MUSIC是强基于方向正交得到的结果，因此其声学图像中**基本不包含声源的功率信息**，所以本来两个强度相差较大的声源在music图中获得了相同峰值。取而代之的就是**弱声源方向清晰了很多**
* 两个声源方向均与上一章存在一点差异，目前不知道原因

# 优缺点

优点：

* **超高分辨率**：这是 MUSIC 的核心优势。只要快拍数足够且信噪比适中，它可以突破瑞利极限，分辨出角度差极小的两个声源。
* **抗噪性能强**：由于将噪声剥离到了独立的子空间中，其算法本身对各向同性的高斯白噪声具有很强的容忍度。

缺点：

* **对相干声源极为敏感**：当在强混响环境中，会导致信号子空间维度丢失，**部分声源特征向量混入噪声子空间**，导致漏检和偏移
* **必须已知声源个数：**必须确认声源个数才能准确划分子空间，不然也会导致声源特征向量混入噪声子空间
* 依赖阵元数量：由算法可知，噪声子空间维度是由阵元数量决定的，阵元越多，声源估计越准，并且阵元数量必须至少大于声源数量

# 总结

MUSIC的核心就是对接收信号协方差矩阵**<u>进行特征值分解，得到噪声子空间</u>**，然后通过导向矢量扫描，寻找投影零点，也就是空间谱极值点。利用的<u>最核心原理是Hermitian矩阵的特征向量必定互相正交</u>
