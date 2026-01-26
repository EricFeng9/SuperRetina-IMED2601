# SuperRetina v6: Dual-Path Hybrid Multimodal Registration
> **Overview**: v6 是一套端到端 (End-to-End) 的多模态配准方案。它摒弃了手动图像预处理（如反色），采用 **双路编码器 (Dual-Path Encoder)** 架构来直接学习跨模态特征映射。为了解决双路网络的特征对齐难题及背景误检问题，引入 **GT Anchor (真值锚点)** 强监督与 **Mask Constraint (掩码约束)** 机制。

---

## 1. 网络架构 (Network Architecture)

### 1.1 Dual-Path Encoder (双路伪孪生网络)
针对多模态图像（如 CF 黑血管 vs FA 白血管）的视觉差异，采用独立的特征提取路径：
*   **Path A (Fix Encoder)**: 
    *   **输入**: Fix Image (e.g., CF, raw intensity).
    *   **职责**: 学习 CF 模态下的血管特征（通常为暗线结构）。
    *   **权重**: 独立更新，不共享。
*   **Path B (Mov Encoder)**:
    *   **输入**: Moving Image (e.g., FA/OCTA, raw intensity).
    *   **职责**: 学习 FA/OCTA 模态下的血管特征（通常为亮线/颗粒结构）。
    *   **权重**: 独立更新，不共享。

### 1.2 Shared Heads (共享多任务头)
尽管编码器独立，但顶层任务头必须共享，以确保证特征空间的统一：
*   **Detector Head**: 输出关键点概率图 (Heatmap)。共享权重。
*   **Descriptor Head**: 输出密集特征描述子 (Dense Descriptors, e.g., 256-d)。共享权重。

---

## 2. 训练策略 (Training Strategy)

### 2.1 GT Anchor Alignment (真值锚点强对齐)
为了强制两个独立的编码器将不同模态的同一解剖结构映射到相同的特征向量：
*   **机制**: 在每个 Training Step，利用算法提取 CF 图像的 **GT 血管分叉点**。
*   **操作**: 无论模型当前检测出什么点，**强制计算 GT 点位置的 Descriptor Loss**。
*   **Loss**: $L_{anchor} = \text{InfoNCE}(Desc_{fix}[GT], Desc_{mov}[GT])$。
*   **效果**: 哪怕模型初期什么都检测不到，Anchor Loss 也会迫使两路编码器迅速对齐特征空间。

### 2.2 Self-Supervised PKE with GT Init (GT初始化的自监督PKE)
采用 PKE (Probabilistic Keypoint Estimation) 进行关键点挖掘，但改变其初始化方式：
*   **Step 0 (Initialization)**: 
    *   将 **GT 分叉点** 作为初始的伪标签 $Y_{init}$。
    *   告诉 PKE: "最开始，这些就是好的关键点。"
*   **Step 1+ (Self-Discovery)**: 
    *   PKE 基于当前的特征空间，寻找几何一致性高的新点 (New Candidates).
    *   更新伪标签: $Y_{new} = Y_{prev} \cup \text{Candidates}$ (经过价值过滤)。
*   **优势**: 解决了 PKE冷启动盲目搜索的问题，同时赋予模型发现 Mask 外（如果存在）或 Mask 未标注的细微特征的能力。

### 2.3 Mask-Constrained Detector Loss (掩码约束检测)
为了根除背景误检（如眼球反光、噪点），对 Detector 施加硬性约束：
*   **原理**: 任何落在血管掩码 (Vessel Mask) 之外的高置信度检测，都是错误的。
*   **Loss 设计**:
    $$ L_{det} = L_{main} + \alpha \cdot L_{suppression} $$
    *   $L_{main}$: 拟合 PKE 挖掘出的正样本点 (Dice Loss).
    *   $L_{suppression}$: 抑制背景区域响应。
        $$ L_{suppression} = \text{Mean}(Heatmap \cdot (1 - Mask)) $$
    *   若模型在 Mask 外预测了热度，该项 Loss 会急剧升高。

---

## 3. 实施流程 (Implementation Workflow)

### 3.1 数据准备
*   **Input**: 原始 CF, FA/OCTA 图像 (不做反色, 不做直方图匹配).
*   **Labels**: 
    1.  **Vessel Mask**: 用于背景抑制。
    2.  **GT Keypoints**: 实时从 Mask 提取，用于 Anchor Alignment 和 PKE Init。

### 3.2 损失函数总览
$$ L_{total} = \lambda_{det} L_{det} + \lambda_{desc} L_{desc} $$
*   **Descriptor Loss**: 包含 PKE 挖掘点对 + GT Anchor 点对。重点加权 GT Anchor。
*   **Detector Loss**: 包含 PKE 伪标签拟合 + 背景抑制 (Mask Constraint)。

### 3.3 超参数设置
*   **Augmentation**: 
    *   Detector Rotation: $[-30, 30]$ (找点鲁棒性)
    *   Descriptor Rotation: $[-10, 10]$ (特征一致性)
*   **Weights**:
    *   $\alpha_{suppress}$: 1.0 ~ 5.0 (背景抑制权重)
    *   $\lambda_{anchor}$: 10.0 (强行对齐权重)

---

## 4. 预期里程碑
1.  **Epoch 0-10**: Loss 下降，匹配数极少（特征未对齐）。
2.  **Epoch 10-20**: 匹配数激增，且全部集中在血管上（GT Anchor 生效，Mask 抑制背景）。
3.  **Epoch 50+**: 匹配精度进一步提升，能够处理轻微形变和旋转。
4.  **Final**: 在真实多模态数据上，不仅能匹配分叉点，还能对齐细微血管结构。
