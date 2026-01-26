# SuperRetina v4 完整实施方案: 双路编码器 (Dual-Path Encoder)

## 1. 核心改进背景
在多模态眼底图像配准（尤其是 **CF vs FA/OCTA**）中，即使是同一解剖结构（血管），其视觉表征也存在根本性冲突：
*   **CF (Color Fundus)**: 血管呈**暗色**，背景较亮。
*   **FA/OCTA**: 血管呈**亮色**，背景较暗。

v3 版本使用共享权重的 Siamese 网络，试图强迫同一组卷积核适应这种“明暗反转”的特征，导致模型难以收敛到血管中心，而是倾向于识别梯度最大的血管壁边缘，造成**取点偏移 (Off-center Keypoints)**。

**v4 方案核心：** 放弃完全共享权重的 Siamese 架构，采用 **Pseudo-Siamese (伪孪生)** 架构，即**双路独立编码器 + 共享解码器头部**。

---

## 2. 网络架构设计 (Pseudo-Siamese)

网络分为两条独立的编码路径，但在高层语义和输出层进行强制对齐。

### 2.1 双路编码器 (Dual-Path Encoders)
*   **Encoder_Fix (针对 CF 模态)**
    *   **功能**: 专门处理固定图像（通常为 CF）。
    *   **特性**: 独立学习“从亮背景中提取暗细线条”的能力。
    *   **权重**: 独立更新，不与 Moving 分支共享。
*   **Encoder_Mov (针对 FA/OCTA 模态)**
    *   **功能**: 专门处理运动图像（通常为 FA 或 OCTA）。
    *   **特性**: 独立学习“从暗背景中提取亮细线条”的能力。
    *   **权重**: 独立更新。

### 2.2 共享头部 (Shared Heads)
为了保证两条路径提取的特征在最终的语义空间是**可比较**和**可匹配**的，解码器部分必须共享：
*   **Descriptor Head (共享)**:
    *   将 Encoder_Fix 和 Encoder_Mov 提取的特征图映射到同一个 256维 Metric Space。
    *   通过 InfoNCE Loss 强制同名点在特征空间距离拉近。
*   **Detector Head (共享)**:
    *   输出关键点概率图 (Probability Map)。
    *   强制“血管分叉点”的定义在两种模态下一致。

---

## 3. 完整训练流程

### 3.1 数据输入与预处理
*   **输入对**: $I_{fix}$ (CF) 和 $I_{mov}$ (FA/OCTA)。
*   **域随机化 (Domain Randomization)**:
    *   分别对两张图独立施加亮度不均 (Bias Field)、斑点噪声 (Speckle Noise) 等，打破纹理依赖。
*   **GT 准备**:
    *   **血管掩码 (Vessel Mask)**: 仅用于生成初始监督信号（提取分叉点），**不**作为网络输入。
    *   **几何变换 ($H_{gt}$)**: 数据加载器生成的真值单应性矩阵。

### 3.2 前向传播 (Forward Pass)
1.  **Fix 分支**: $I_{fix} \to \text{Encoder\_Fix} \to \text{Feat}_{fix}$
2.  **Mov 分支**: $I_{mov} \to \text{Encoder\_Mov} \to \text{Feat}_{mov}$
3.  **Head 输出**:
    *   $\text{Feat}_{fix} \to \text{Shared Heads} \to (\text{Det}_{fix}, \text{Desc}_{fix})$
    *   $\text{Feat}_{mov} \to \text{Shared Heads} \to (\text{Det}_{mov}, \text{Desc}_{mov})$

### 3.3 损失函数设计
*   **检测损失 ($L_{det}$)**: 使用 Dice Loss。
    *   初期使用 GT 血管分叉点热力图监督。
    *   后期使用 PKE 自监督热力图监督。
*   **描述子损失 ($L_{desc}$)**: 使用 **InfoNCE Loss**。
    *   利用 $H_{gt}$ 建立像素级对应关系。
    *   将 Batch 内所有非对应点作为负样本，防止特征坍塌。

---

## 4. 三阶段课程学习 (Curriculum Learning)

为了验证双路编码器的核心有效性，我们暂时**移除所有复杂的课程学习逻辑**（如渐变权重、几何一致性过渡期等），仅保留最基础的“冷启动”以防止训练崩溃。

| 阶段 | 周期 | 任务 | 逻辑 |
| :--- | :--- | :--- | :--- |
| **Stage 1: Warmup** | Epoch 1-5 | 描述子对齐 | 因双路编码器互不共享，初始特征空间完全未对齐。需利用 GT H 和 InfoNCE Loss 快速拉近两路特征，否则 PKE 无法工作。此时关闭 PKE。 |
| **Stage 2: Standard** | Epoch 6+ | 标准训练 | 开启 PKE。完全沿用标准 SuperRetina 逻辑，不再区分特殊阶段。 |

---

## 5. 推理流程 (Inference)
在实际测试时，流程如下：

1.  **加载模型**: 加载训练好的 `Encoder_Fix`, `Encoder_Mov` 和共享 Heads。
2.  **输入分流**:
    *   CF 图像 $\to$ `Encoder_Fix`路径 $\to$ 关键点 & 描述子。
    *   FA/OCTA 图像 $\to$ `Encoder_Mov`路径 $\to$ 关键点 & 描述子。
3.  **特征匹配**:
    *   使用 descriptor 进行 Nearest Neighbor 匹配。
    *   Lowe's Ratio Test 过滤误匹配。
4.  **几何解算**:
    *   RANSAC 求解最终单应性矩阵 $H$。

## 6. 代码文件结构
*   **`train_multimodal_v4.py`**: 主训练脚本，强制 `shared_encoder=False`。
*   **`model/super_retina_multimodal.py`**: 定义了 `SuperRetinaEncoder` 类和含分支逻辑的 `SuperRetinaMultimodal` 类。
*   **`config/train_multimodal.yaml`**: 配置文件。