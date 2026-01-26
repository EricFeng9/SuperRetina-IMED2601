# SuperRetina v4.1 方案: Shared Encoder + Inversion + Supervised PKE

## 1. 核心反思与调整
经过 v4.0 (Dual-Path) 的实验，我们发现：
1.  **Dual-Path 的问题**：双路编码器虽然解决了明暗反转问题，但导致特征空间割裂，描述子难以对齐，在大角度旋转或复杂变换下匹配失效 (MACE 极高)。
2.  **v3 的优势**：Shared Encoder (Siamese) 配合手动反色 (Inversion)，利用了卷积网络对相似纹理处理的一致性，描述子匹配性能更稳健。
3.  **v4 的亮点**：Supervised PKE (强监督注入) 能有效引导检测器 (Detector) 关注血管分叉点，解决“取点偏离中心”的问题。

**v4.1 决策**：**回归 v3 架构 (Shared Encoder + Inversion)，但引入 v4 的 Supervised PKE 训练策略。**

---

## 2. 网络架构 (回归 Siamese)

*   **Shared Encoder**: 恢复使用 **共享权重** 的编码器。
*   **输入预处理**: 
    *   **Fix (CF)**: 血管为暗色。
    *   **Mov (FA/OCTA)**: 血管为亮色。
    *   **策略**: 对 CF 图像进行 **反色 (Invert)** 处理 `1.0 - img`，使其血管变亮，从而与 FA/OCTA 在视觉特征上对齐。
*   **Head**: 共享 Detector 和 Descriptor Head。

---

## 3. 训练策略 (融合 v4 特性)

采用 **Supervised PKE** 策略，解决取点不准问题。

| 阶段 | 周期 | 策略 | 详细逻辑 |
| :--- | :--- | :--- | :--- |
| **Stage 1: Supervised PKE** | Epoch 1-20 | **强监督预热** | 1. **输入注入**: 将 GT 血管分叉点直接喂给模型。<br>2. **Loss**: 计算 Dice Loss (Pred vs GT_Heatmap)。<br>3. **目的**: 强迫 Shared Encoder 在处理反色后的 CF 图时，学会忽略背景噪声，精准定位分叉点。 |
| **Stage 2: Standard PKE** | Epoch 21+ | **自适应微调** | 1. **PKE**: 开启自监督挖掘逻辑。<br>2. **Loss**: 让模型在特征空间中自我迭代，进一步提升匹配鲁棒性。 |

---

## 4. 关键参数设置
*   **Descriptor Augmentation**: 回归小角度旋转 `[-10, 10]`，避免强求 CNN 学习 30 度旋转不变性导致特征坍塌。
*   **Shared Encoder**: `True`.
*   **Inversion**: `True` (对 CF 进行反色)。

---

## 5. 预期效果
*   **匹配率 (MIR)**: 应恢复到 v3 的高水平。
*   **取点精度**: 得益于 Stage 1 的强监督，应比 v3 更准确（落在血管中心）。
*   **收敛速度**: 由于是 Shared Encoder，收敛应比 Dual-Path 快得多。