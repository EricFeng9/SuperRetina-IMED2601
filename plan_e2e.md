# 端到端多模态配准训练计划 (Plan E2E)
我现在在做一个课题，目标是目标是基于我们已有的生成数据集（结构完全相同且对齐的cf-oct-fa图像对）训练出一个支持cf-fa，cf-oct（cf均为fix）的多模态配准模型。我们目前在用fractMorph模型进行2D化改造后的模型，但是fractMorph2D只能针对一些微小血管形变进行对齐和修复，但是针对真实情况不对齐图像中出现的血管大尺度旋转（如15 30度的旋转角度）、大的位置偏移的情况下配准效果较差，1是原先的逐像素点移动的模式很差2是我们修改成让模型直接预测放射参数，效果也很差，几乎不收敛。我现在的想法是改造SuperRetina，让模型学习cf图像和oct/fa图像之间的关键点匹配，然后利用这些匹配出来的关键点计算放射矩阵，对跨模态图像进行配准
真实数据集没有血管分割图，原图没对齐，但是有人工标注的关键点用于计算仿射矩阵
生成数据集有血管分割图，且原图是结构完全一致的
## 核心目标
将原有的分阶段训练（Phase 1: Warmup -> Phase 2: Geo -> Phase 3: PKE）改为**端到端联合训练**。
**动机**：用户怀疑分阶段训练导致模型在 Phase 1 过拟合了生成数据的特定风格，导致后期迁移到真实数据时失效。端到端训练允许检测器和描述子在同一优化过程中相互适应，避免单一模块过早陷入局部最优。

## User Review Required
> [!IMPORTANT]
> 本计划将完全移除 Phase 1/2/3 的逻辑判断，所有 Loss 从 Epoch 1 开始同时作用。
> 这可能会导致训练初期不稳定，建议密切关注前 50 个 Epoch 的 loss 曲线和可视化结果。

## Proposed Changes

### 1. Training Script (`train_multimodal_e2e.py`)
- **[NEW]** 创建新脚本 `train_multimodal_e2e.py`（基于 `train_multimodal_v3.py`）。
- **逻辑变更**：
  - 移除 `if epoch <= 20...` 的阶段判断逻辑。
  - 始终设置 `model.PKE_learn = True`。
  - 始终计算 `loss_det` (GT监督), `loss_des` (GT Triplet), `loss_geo` (几何一致性), `loss_pke` (PKE自监督)。

### 2. Model (`model/super_retina_multimodal.py`)
- **[MODIFY]** 修改 `forward` 函数或增加一个新的 `forward_e2e` 方法。
- **Loss 构成 (Joint Loss)**：
  $$L_{total} = \lambda_{det} L_{det} + \lambda_{des} L_{des} + \lambda_{geo} L_{geo}$$
  - **$L_{det}$**: `Dice(pred, gt_heatmap)`。使用 GT 血管分叉点生成的 Gaussian Heatmap 进行强监督。
  - **$L_{des}$**: `TripletLoss(anchor, pos, neg)`。利用 GT Homography ($H_{0\to1}$) 寻找正样本对，确保描述子对齐。
  - **$L_{geo}$**: `Dice(det_fix, Warp(det_mov))`。利用 GT Homography 约束检测器的几何一致性。
  - **PKE Loss**: **始终开启**。PKE 是 SuperRetina 的核心，负责挖掘非血管的关键点。即便在早期，我们也启用 PKE 机制，但依靠强监督 ($L_{det}$ 和 $L_{des}$) 来快速引导模型收敛，防止 PKE 在初始阶段“瞎猜”。
    - **策略调整**: 在训练初期 (e.g. Epoch 1-5)，虽然开启 PKE，但由于模型未收敛，我们会通过较高的阈值或 Value Map 机制自然过滤掉大部分不可靠点；随着 $L_{det}$ 拉高基础能力，PKE 生成的伪标签会逐渐增多且变准。
    - **Value Map**: 从 Epoch 1 就开始维护，记录关键点的历史可靠性。

### 3. Execution Steps
1. 复制 `train_multimodal_v3.py` 为 `train_multimodal_e2e.py`。
2. 移除 Phase 逻辑，统一 Loss 计算。
3. 在 `model` 中确保支持这种混合调用（目前的 `forward` 比较耦合 Phase，可能需要清理一下）。

## Verification Plan

### Automated Verification
- 运行 `python train_multimodal_e2e.py --epoch 5` 进行冒烟测试，确保 Loss 下降，无报错。

### Manual Verification
- 检查 `validation_log.txt`，确认 MSE 在下降。
- 检查 `save/.../epochX/sample_ID_matches.png`，确认在 Epoch 5 左右就能看到合理的连线（端到端通常收敛较快）。
- 对比 `fix_registered_checkerboard.png` 确认配准效果。
