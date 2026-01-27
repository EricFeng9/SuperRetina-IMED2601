# SuperRetina v7: Pretrained-Driven Fine-Tuning + Dual-Path Multimodal Alignment

> **Overview**: v7 方案在 v6 的基础上引入 **官方预训练权重** 进行微调。通过 **双路独立初始化** 继承解剖结构感知能力，利用 **Phase-wise Freezing (分阶段冻结)** 和 **Layer-wise LR (分层学习率)** 解决亮/暗血管极性反转导致的跨模态匹配难题。

---

## 1. 网络架构 (Network Architecture)

### 1.1 Dual-Path Encoder (双路编码器)
*   **Path A (Fix Encoder)**: 处理 CF 图像（暗线特征）。**初始化**: 加载官方权重。
*   **Path B (Mov Encoder)**: 处理 FA/OCTA 图像（亮线特征）。**初始化**: 同样加载官方权重（继承结构理解能力）。
*   **权重**: 独立微调，不共享。

### 1.2 Shared Heads (共享多任务头)
*   **Detector Head**: 共享。**初始化**: 加载官方权重，初始阶段冻结权重。
*   **Descriptor Head**: 共享。**初始化**: 加载官方权重，作为 Metric Space 统一的基础。

---

## 2. 课程学习策略 (Curriculum Learning)

### 2.0 Phase 0: Modality Adaptation & Space Alignment (模态适配与空间对齐)
*   **目标**: 让模型学会在保留结构感知的同时，将不同极性（亮/暗）的特征映射到同一分布。
*   **操作**:
    1.  **输入**: 使用全对齐图对。
    2.  **冻结策略**: **冻结 Detector Head**，仅训练 Encoder 和 Descriptor Head。
    3.  **学习率**: Encoder 使用低学习率 ($10^{-5}$)，Header 使用正常学习率 ($10^{-4}$)。
    4.  **Loss: 高效密集 InfoNCE**: 
        *   在此阶段通过对比学习强制让 Moving 支路产生的描述子向 Fix 支路（即官方权重定义的标准分布）靠拢。
*   **关键**: 保护预训练好的检测器响应不被初期跨模态噪声破坏，只通过描述子损失“拉齐”两端特征。

### 2.1 Phase 1+ : Progressive Keypoint Expansion (PKE)
*   **目标**: 加入几何畸变，寻找可重复的关键点。
*   **操作**:
    1.  引入随机仿射变换 ($T$)，输入 `fix` 和 `moving_warped`。
    2.  启用 PKE 流程，以 **GT 血管分叉点** 为种子进行自主挖掘。
    3.  描述子 Loss 回归原始 SuperRetina 逻辑（基于采样点对的 Triplet Loss）。

---

## 3. 核心约束机制 (Core Constraints)

### 3.1 Symmetric Mask Constraint (对称二向掩码抑制)
为了解决 Moving 分支在背景区域产生垃圾点导致的匹配干扰：
*   **Fix 支路**: 使用原始血管 Mask 抑制背景。
*   **Moving 支路**: 利用真值矩阵 $H_{0to1}$ 将血管 Mask 投影 (Warp) 到 Moving 空间。
*   **Loss**: 
    $$ L_{suppress\_all} = \text{Mean}(Det_{fix} \cdot BG\_Mask_{fix}) + \text{Mean}(Det_{mov} \cdot BG\_Mask_{mov}) $$ 
*   **效果**: 确保双路出来的候选关键点 **全部** 落在血管上，极大降低匹配阶段的搜索噪声。

### 3.2 描述子损失回归 (Reverting to Original SR Loss)
*   **摒弃掉 v6.1 中的 InfoNCE 超高权重逻辑**。
*   使用原版 SuperRetina 的描述子相关性损失，保证特征分布的稳定性。
*   保留 GT-Init PKE 策略，作为种子点引导。

---

## 4. 损失函数 (Loss Functions)

$$ L_{total} = L_{det} + L_{desc} + \alpha \cdot L_{suppress\_all} $$

*   $L_{det}$: PKE Dice Loss (拟合演化后的伪标签)。
*   $L_{desc}$: 采样点对的像素级关联损失 (原版 SR 逻辑)。
*   $L_{suppress\_all}$: 双边背景抑制损失（权重下调至 0.1-0.2，防止坍缩）。

---

## 5. 预期里程碑 (Expected Milestones)

1.  **Phase 0 完成后**: 模型能够识别出跨模态下的同名点。特征对齐度极高（PCA 可视化颜色一致）。
2.  **Phase 1 开始后**: 开放检测头微调，检测器在预训练的基础上，迅速适应跨模态下的血管分叉响应。
3.  **微调优势**: 相比从零训练，收敛速度提升 50% 以上，对不同亮标/模糊程度的图像鲁棒性显著增强。

