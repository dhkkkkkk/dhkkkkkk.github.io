---
title: 声源定位之基于相对时延估计
date: 2026-06-04 16:03:39
tags: 麦克风阵列算法
mathjax: true
---

我是后背

要想实现一个声源的定位，我们需要：

- 声源的到达**方向** (Direction of Arrival，DOA) ，主要计算其的方位角和俯仰角
- 声源的**距离**估计

目前我们的重点先放在DOA上，对于DOA估计，主要有三个方法：

* 基于**到达时间差（TDOA）**：计算不同阵元之间接收到的信号的时延差，根据该时延差即可推断出声源位置
* 基于**波束形成**的方法：进行不同角度遍历扫描，分别进行波束形成（即假设时延差），功率最大处即为声源位置
* 基于高分辨谱估计：然后采用特征分解操作，从协方差矩阵中提取出信号子空间。之后，利用空间谱 估计技术对该子空间进行分析，最终得出声源方向的估计结果。

根据定义我们就可以看出，基于相对时延估计的技术点主要在于如何计算出时延差；而波束形成方法则更简单暴力，技术点在于使用什么波束形成方法更好

本章我们将对基于相对时延估计的方法进行展开，其可以分为两部分：

* 时间差计算
* 角度计算

其中，时间差计算为核心

# 声场模型

根据麦克风阵列和声源距离的远近，我们可以将声场模型分为：**<u>近场模型</u>和<u>远场模型</u>**，在近场模型下我们将声波看作为**球面波**，其主要考虑麦克风阵列各阵元接收到信号的幅度差; 而远场模型将声波看作为**平面波**，忽略阵元接收信号间的幅度差，近似认为各阵元接收信号之间为简单的时延关系

评判模型的标准为声源到阵列参考阵元的瑞利距离：
$$
R_r = \frac{2D^2}{\lambda}
$$

* D：阵列最大几何尺寸
* λ：波长（传播速度/频率）

当绝对距离大于该值时即为远场模型

由公式可知，**<u>当选择的频率越大，瑞利距离越大</u>**，即该频率越容易表现出近场特性，因此**若要使用更简单的远场模型，我们选择的频率应该偏小一点，**工程中常选取主要频带上的最大频率值

例子：当我们阵列直径为0.2m时，瑞利距离则约为0.47m

如果对近场模型使用了基于远场模型的算法，则会导致**导向矢量的失配**，最终导致结果的不准确

# 传统互相关法

假设参考阵元信号为x1，则我们可以直接通过互相关函数计算阵元x2与x1的时延差：
$$
R_{x_1 x_2}(\tau) = \int_{-\infty}^{\infty} x_1(t) x_2(t + \tau) dt
$$
当互相关值最大时，对应的tau即为时延差。然而，在实际室内环境中，由于声波反射（混响）和环境噪声的影响，**传统互相关函数的峰值往往会变得非常平缓，甚至出现多个伪峰**，导致估计彻底失效。

# 广义互相关法GCC

由维纳-辛钦定理可知，随机信号的**自相关函数和功率谱密度函数服从一对傅里叶变换的关系**（即互相关时域函数在频域的表达就是互功率谱密度），**功率谱密度函数**可以表示为：
$$
P(ω)= X_1(ω) \cdot X_2^*(ω)=\int_{-\infty}^{\infty} R(\tau)e^{-jω\tau} d\tau
$$
这里我们就建立了频域表达与互相关函数（时域）的等式。为了增强互相关函数峰值，我们<u>引入一个加权函数</u>ψ来**增强信号中有用的<u>频率部分</u>**，在频域增强完后再转到时域：
$$
R_{x_1 x_2}^{GCC}(\tau)=\int_{-\infty}^{\infty}\psi(ω) P(ω)e^{jω\tau} dω
$$
对于这个加权函数的设计，则就是GCC各种分支的由来了

## 相位变换加权（PHAT,phase transform）

互功率谱密度是指在f处的共有的能量密集度和在f下两信号的相位差，而我们现在**只需要通过相位差来得到时延值**，因此，可以直接令加权函数为**功率谱密度的<u>模（幅值）</u>**的倒数：
$$
\psi_{PHAT}(f) = \frac{1}{|P_{x_1 x_2}(f)|} = \frac{1}{|X_1(f) \cdot X_2^*(f)|}
$$
这样的话，互相关函数就是一个**仅与<u>功率谱密度的相位</u>相关**的函数了：
$$
R_{x_1 x_2}^{PHAT}(\tau) = \int_{-\infty}^{\infty} e^{j \theta(f)} e^{j 2 \pi f \tau} df= \int_{-\infty}^{\infty} e^{j 2 \pi f (\tau - \tau_0)} df
$$
也就是说，PHAT本身就是指通过相位差来计算时延值，当t取到t0时，互相关函数就是最大值

## py实现

```py
def gcc_get_tdoa(self, freq_range=None):

    if freq_range is None:
        freq_range = [1000, 2000]
    mask = (self.freqbin >= freq_range[0]) & (self.freqbin <= freq_range[1])

    X_ref = self.data[0, 0:1, :]
    G = np.conjugate(X_ref) * self.data[0,:,:]
    eps = 1e-12
    G_phat = G / (np.abs(G) + eps)
    G_filtered = G_phat * mask[None, None, :]   #带通滤波
    G_avg = np.mean(G_filtered, axis=0)
    gcc_time = np.fft.irfft(G_avg, n=self.num_freqs, axis=-1)

    gcc_time_shifted = np.fft.fftshift(gcc_time, axes=-1)

    #构建中心对齐的时延轴（单位：秒）
    lags = np.arange(-self.num_freqs // 2, self.num_freqs // 2) / self.sampling_frequency

    # 9. 寻找互相关谱峰对应的索引，提取 TDOA
    # tdoa_indices 形状: (Nchan,)
    tdoa_indices = np.argmax(gcc_time_shifted, axis=-1)
    tdoa = lags[tdoa_indices]

    return tdoa, gcc_time_shifted, lags
```

这里简单实现了基于麦克风第一个通道的GCC-PHAT计算，这里需要注意几点：

* `np.conjugate(X_ref)`：对参考通道进行共轭转置，及对应上面的公式中的-t0，这样晚于参考通道的信号的t就是正值
* 带通滤波：若要进行窄带运算，必须要使用带通滤波，而不是简单的仅提取对应通道，因为后面要使用ifft
* `gcc_time_shifted`：当ifft对象为普通信号时，其会正常还原时域波形；但如果对象为互功率谱，其对应的时域为互相关函数，包含正负时延，因此ifft的输出会首先输出正时延值，再输出负时延值（与常规顺序相反），gcc_time_shifted会根据0值位置，将正负互换
* `lags = np.arange...`刚刚提到，互相关函数的完整物理定义域是有正负的，具体其实就是[-0.5T,0.5T]（因为任何傅里叶变换得到的都是一个周期函数/序列，ifft得到的就是各频段周期函数的叠加，而一个长度为N的周期序列，任何索引n都可以映射到唯一区间[-0.5T,0.5T]），因此gcc_time_shifted**对应的时间索引是一定对称的**，而lags就是把这个索引值算出来，因为gcc_time_shifted没有包含每个索引对应的时延值，<u>它代表的是每个时延值上对应的互相关值</u>

## 优缺点

优点：由于仅通过相位差来得到时延值，因此该方法拥有极强的抗混响能力（混响对相位影响小）

缺点：当信号中存在噪声时，当某些频段上噪声为主要成分时，也就意为着此时得到的相位也是噪声的相位，而PHAT舍弃了幅值，因此也将**放大噪声**，最终导致互相关函数上真**峰降低**，并出现**大量伪峰**（因为互相关函数代表的是不同t上信号的相关程度，伪峰即为噪声与信号产生的相关）

后记，由于是对每个频段分别进行相位差计算，因此在噪声源存在的情况下几乎没办法准确定位声源（即使噪声比较微弱）

# SRP-PHAT（**Steered Response Power with Phase Transform**）

准确来说这个方法已经不算是直接计算DOA，但是跟GCC-PHAT强相关，所以也放这里了

SRP-PHAT**先在空间中划分网格（候选声源点）**。对于空间中任意一个假设的声源点，计算它到达所有麦克风对的理论时延。然后将所有麦克风对在这些理论时延处的 GCC-PHAT 值进行**空间累加**。

这和基于波束形成的方法非常像，只是波束形成的输出是声压，而SRP-PHAT的输出是互相关函数值：
$$
P(\mathbf{s}) = \sum_{i=1}^{M} \sum_{j=1}^{M} R_{x_i x_j}^{PHAT}\left( \tau_{ij}(\mathbf{s}) \right)
$$
显而易见，该方法要计算任意两个阵元的互相关值，计算量极大，即使预先计算出所有的时延，仍要计算大量次互相关，因此实际使用时一般配合一些简化计算量的优化：

**由粗到精的搜索（Coarse-to-Fine Search）**：先用极稀疏的网格扫描空间，锁定高能量的局部区域；再在这些候选区域内细化网格进行微观搜索。

**随机区域收缩法（Stochastic Region Contraction, SRC）**：基于动态优化思想，每次随机抽取空间样本点，根据响应功率不断收缩搜索空间收敛到最优解，能将计算量降低几个数量级。

**GPU 并行加速**：由于每个空间网格点的功率计算 彼此完全独立，属于典型的数据并行（SIMD）任务。在现代工程中，利用 CUDA 将网格映射分发给 GPU 执行，可以极其轻松地实现高分辨率的三维实时 SRP-PHAT 定位。

## 代码实现：

### 建立麦克风对tdoa

由于pair tdoa是srp-phat的主要计算量处，而其结果仅与阵列坐标和扫描角度有关，因此通常是预计算，在后续计算中直接读取保存的toda结果

```python
def build_pair_tdoa_table(
        mic_struct,
        phi_scan,
        theta_scan,
        sound_speed=343.0):
    #计算每个麦克风对的时延（n*n-1 /2）
    M = mic_struct.shape[0]

    pairs = list(combinations(range(M), 2))
    Npair = len(pairs)

    pair_tdoa = np.zeros(
        (
            len(theta_scan),
            len(phi_scan),
            Npair
        ),
        dtype=np.float32
    )

    for it, theta in enumerate(tqdm(theta_scan,desc="计算各麦克风pairs时延中")):

        sin_t = np.sin(theta)
        cos_t = np.cos(theta)

        for ip, phi in enumerate(phi_scan):

            u = np.array([
                sin_t*np.cos(phi),
                sin_t*np.sin(phi),
                cos_t
            ])

            tau = -mic_struct @ u / sound_speed

            for k, (i, j) in enumerate(pairs):

                pair_tdoa[it, ip, k] = (
                    tau[i] - tau[j]
                )#注意这里tau是带了-号，如果i是-1，j是-2，则结果+1则代表i晚于j 1s

    pair_lag = np.round(pair_tdoa * 51200).astype(np.int32)
    np.savez_compressed(
        "pair_tdoa.npz",
        pair_tdoa=pair_tdoa,
        pairs=pairs,
        pair_lag = pair_lag,
        phi_scan=phi_scan,
        theta_scan=theta_scan,
    )

    return 1
```

我们后续使用的主要是pair_lag（每对时延值对应的互相关函数采样索引，后面再说详细情况）和pairs（麦克风对索引）

### 计算麦克风对的互相关函数

对每个麦克风对的信号进行互相关计算

```python
def compute_gcc_phat(spec, pairs, N_fft):

    eps = 1e-12

    Npair = len(pairs)

    gcc = np.zeros(
        (Npair, N_fft),
        dtype=np.float32
    )

    for k, (i, j) in enumerate(tqdm(pairs, desc="gcc计算中")):

        cross = spec[:,i,:] * np.conj(spec[:,j,:])

        cross /= np.abs(cross) + eps
        avg_cross = np.mean(cross,axis=0)
        gcc_time = np.fft.irfft(avg_cross)

        gcc_time = np.real(
            np.fft.fftshift(gcc_time)
        )

        gcc[k] = gcc_time

    return gcc #(Npair,F)
```

得到是每个麦克风对对应的互相关函数

这个函数需要注意，输入的spec如果是单边谱，则使用irfft还原，如果是双边谱，就用ifft。如果这里没有对应，最后输出的功率谱上声源方向会出现缩放情况

### SRP-PHAT扫描

```python
def srp_phat_fast(
        gcc,
        pair_lag):

    Npair, Nlag = gcc.shape

    center = Nlag // 2

    idx = center + pair_lag

    idx = np.clip(
        idx,
        0,
        Nlag - 1
    )

    gcc_expand = gcc[
        np.arange(Npair)[None, None, :],
        idx
    ]

    power_map = np.sum(
        gcc_expand,
        axis=2
    )

    return power_map
```

之前得到的pair_lag是根据0点计算的索引，也就是一个带正负的索引，所以在用之前要先进行`idx = center + pair_lag`转化为真正的[0,N_fft]的索引，用于索引gcc

SRP-PHAT就是直接用假想方向的tdoa来索引gcc，使用客观索引避免了噪声干扰，再通过每个麦克风对求和来增加稳定性；gcc-phat就是直接找gcc的最大值，因此更容易被噪声干扰

个人实际使用下来，SPR-PHAT通过查表方式和DAS的计算时长都还不错，接近的计算时间下SPR-PHAT精度甚至更高

DAS成像：

{% asset_img DAS.png This is an image %} 

SRP-PHAT成像：

{% asset_img SRP.png This is an image %} 

# 从TDOA得到DOA

本节默认在远场模型下计算（远场模型只能得到DOA，无法得到距离，想象声源在参考阵元射出的一条射线方向，而波就是以该射线为法线的平面）

先从二维说起，对于两个距离为d的阵元，对于平面波的入射角（与阵元连线的夹角），有：
$$
\Delta x = c \cdot \tau = d \cdot \cos\theta
$$
c为声速，t为TODA，即可通过反三角求解θ

现在来到三维，方向就变为了方位角Φ和俯仰角θ(与xy平面夹角），与球坐标系定义相同，此时声源方向单位向量就可以表示为：
$$
n=[\cos\theta \cos\phi, \cos\theta \sin\phi,\sin\theta]
$$
于是就有：
$$
Δx\cos\theta \cos\phi + Δy\cos\theta \sin\phi + Δz\sin\theta = -c\tau
$$
理论上只需要3阵元（也就是3对TDOA）就可以求解这个方程，当阵元大于3时，一般使用最小二乘法求最优解，对于平面阵列，有：
$$
\mathbf{A} = \begin{bmatrix} \Delta x_{12} & \Delta y_{12} \\ \Delta x_{13} & \Delta y_{13} \\ \vdots & \vdots \\ \Delta x_{i j} & \Delta y_{i j} \\ \vdots & \vdots \end{bmatrix}_{P \times 2}, \quad \mathbf{m} = \begin{bmatrix} n_x \\ n_y \end{bmatrix}_{2 \times 1}, \quad \mathbf{Y} = \begin{bmatrix} -c\tau_{12} \\ -c\tau_{13} \\ \vdots \\ -c\tau_{ij} \\ \vdots \end{bmatrix}_{P \times 1}
$$
线性方程组可以紧凑地表示为：
$$
\mathbf{A}\mathbf{m} = \mathbf{Y}
$$
最小二乘法就是让损失函数（Am-b)^2最小，即让该函数对m的偏导为0，化简后则有：
$$
\mathbf{m} = \begin{bmatrix} n_x \\ n_y \end{bmatrix} = (\mathbf{A}^T \mathbf{A})^{-1} \mathbf{A}^T \mathbf{Y}
$$
（上式为最小二乘法通用解），得到方向向量后根据三角函数关系即可得到具体角度

# 总结

基于TDOA的声源定位方法实现原理较为简单，速度也还算可以（SRP不算该方法），核心就是互相关函数和如何设计加权函数，普遍存在的缺点就是极容易被噪声干扰。
