# SuperRetina v7.1: Pretrained-Driven Fine-Tuning + Dual-Path Multimodal Alignment
我现在在做一个课题，目标是目标是基于我们已有的生成数据集（结构完全相同且对齐的cf-oct-fa图像对）训练出一个支持cf-fa，cf-oct（cf均为fix）的多模态配准模型。我们目前在用fractMorph模型进行2D化改造后的模型，但是fractMorph2D只能针对一些微小血管形变进行对齐和修复，但是针对真实情况不对齐图像中出现的血管大尺度旋转（如15 30度的旋转角度）、大的位置偏移的情况下配准效果较差，1是原先的逐像素点移动的模式很差2是我们修改成让模型直接预测放射参数，效果也很差，几乎不收敛。我现在的想法是改造SuperRetina，让模型学习cf图像和oct/fa图像之间的关键点匹配，然后利用这些匹配出来的关键点计算放射矩阵，对跨模态图像进行配准
> **Overview**: v7.1 是在 v7 基础上的 bug-fix 和增强版本。主要解决了 Phase 0 背景抑制逻辑中的漏洞，并引入了正向监督机制。

---

## 1. 关键修复 (Critical Fixes)

### 1.1 Phase 0 "Mask Warping Loophole"
*   **Bug**: 之前在 Phase 0 中，虽然输入图像是 geometrically aligned（几何对齐）的，但 `bg_mask_mov` 错误地使用了随机仿射矩阵 $T$ 进行 Warp。这导致图像未旋转，但抑制用的背景掩码旋转了，进而**错误地抑制了有效血管上的响应**。
*   **Fix**: 在 Phase 0 中强制传入 $H = \text{Identity}$ (或 `None`)，确保背景掩码与图像几何一致。

### 1.2 "Padded Zero" Mask Issue
*   **Bug**: `F.grid_sample` 在 Warp 掩码时，边界外自动填充 `0`。如果 `0` 代表背景，那么这些区域会被正确抑制。但如果掩码定义是 `1=Background`，那么填充 `0` 就是“非背景（血管）”，导致边界外乱飘。
*   **Fix**: 采取 **Inverted Logic**。先 Warp `vessel_mask` (1=血管，0=背景)。这样 padded zeros = 0 = 背景。然后再取反得到 `bg_mask = 1 - warped_vessel`。

---

## 2. 课程学习策略增强 (Phase 0 Enhancement)

### Problem
Phase 0 如果只用 `loss_suppress`（负向抑制），模型可能学会输出全 0 图来最小化损失（Safe Solution），导致进入 Phase 1 时没有检测能力。

### Solution: Positive + Negative Supervision
在 Phase 0 加入**正向激励**：
1.  **Input**: $I_{fix}$, $I_{mov}$ (Aligned).
2.  **GT Heatmap**: 利用已有的 `vessel_keypoints` 生成高斯热力图。
3.  **Positive Loss**: `Dice(Det_fix, GT) + Dice(Det_mov, GT)`。即使是 Phase 0，也要强迫检测头在血管分叉点激活。
4.  **Negative Loss**: `loss_suppress` (Weight = 1.0)。强迫在背景闭嘴。

---

## 3. 损失权重调整
*   `loss_suppress` 权重调整为 **1.0**（原方案中曾设为 0.2 或 2.0）。这是一个平衡值，既能有效抑制，又不至于压倒正向信号。

---

## 4. 可视化验证
*   **Epoch 1 Visualization**: 增加了 `bg_mask_fix` 和 `bg_mask_mov` 的可视化输出，用于人工检查抑制区域是否正确覆盖了背景和黑边。

---

## 5. 预期效果
*   **Phase 0**: Det Loss 不再是 0，而是正值并逐渐下降。
*   **Phase 1**: PKE 启动时，初始种子点质量更高，背景噪声更少。
*   **Result**: 更多的 Matches，更少的 outliers。
