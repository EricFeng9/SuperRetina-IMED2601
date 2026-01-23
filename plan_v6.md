
我现在在做一个课题，目标是目标是基于我们已有的生成数据集（结构完全相同且对齐的cf-oct-fa图像对）训练出一个支持cf-fa，cf-oct（cf均为fix）的多模态配准模型。我们目前在用fractMorph模型进行2D化改造后的模型，但是fractMorph2D只能针对一些微小血管形变进行对齐和修复，但是针对真实情况不对齐图像中出现的血管大尺度旋转（如15 30度的旋转角度）、大的位置偏移的情况下配准效果较差，1是原先的逐像素点移动的模式很差2是我们修改成让模型直接预测放射参数，效果也很差，几乎不收敛。我现在的想法是改造SuperRetina，让模型学习cf图像和oct/fa图像之间的关键点匹配，然后利用这些匹配出来的关键点计算放射矩阵，对跨模态图像进行配准
# 跨模态 SuperRetina 详细训练计划 (Plan v5)

目标：
1. **热身期 (Phase 1)**：既把 **Descriptor** 练好 (利用 GT H + GT 分叉点)，又把 **CF Detector** 拉到“围着 GT 分叉点亮”的状态。
2. **几何一致性预热 (Phase 2)**：在 PKE 开启前，利用 True H 把 **Moving Detector** 带起来，避免一开始就两边一起学歪。
3. **PKE 联合训练 (Phase 3)**：在基础打好后，开启 PKE 自监督，但需严格控制质量。

---

## 阶段划分与实施细则

### 零、基础设施 (Phase 0)
*已就绪*
- **数据层** (`MultiModalDataset`)：
  - `image0` (CF, fixed)
  - `image1` (FA/OCT, moving, random affined)
  - `T_0to1` (True Homography from image0 to image1)
  - `vessel_mask0` (CF vessel mask GT)
- **现有几何逻辑**：
  - Descriptor Warmup 利用 `H_0to1` 约束 GT 对。
  - 联合期利用 `H_0to1` 构造 grid。

---

### 一、阶段 1：热身期 (Phase 1) —— Descriptor + CF Detector 联合热身
**周期**：Epoch 1 – 20
**状态**：PKE **OFF**

#### 1.1 目标
- **Descriptor**：利用 `H_0to1` 和 `vessel_keypoints` (分叉点) 学习鲁棒特征。
- **CF Detector**：受到强监督，学会仅在血管分叉点处高响应，背景处抑制。
- **Moving Detector**：暂不强行约束 (避免 noisy warp)，由 descriptor 分支隐式带动或阶段 2 处理。

#### 1.2 具体改动
- **Input**: `model(..., descriptor_only=True, vessel_keypoints=GT_kps, ...)`
- **Loss 计算** (在 `descriptor_only` 分支内新增):
  1.  **Generate Soft Label**:
      $$Y^{fix} = \text{Gaussian}(vessel\_keypoints)$$
      使用 `self.kernel` 生成高斯热力图。
  2.  **Detector Loss**:
      $$l_{det\_warm} = \text{Dice}(det\_fix, Y^{fix})$$
  3.  **Total Warmup Loss**:
      $$L = \lambda_{det} \cdot l_{det\_warm} + \lambda_{des} \cdot l_{des\_warmup}$$
      建议 $\lambda_{det}=1.0, \lambda_{des}=1.0$。

#### 1.3 预期观察
- `loss_det` 不再为 0，应呈下降趋势。
- `fixed_kps.png` (绿色预测点) 应高度重合于 `vessel_keypoints_extracted.png` (红色 GT 点)。

---

### 二、阶段 2：几何一致性预热 (Phase 2) —— Moving Detector Alignment
**周期**：Epoch 21 – 40
**状态**：PKE **OFF**

#### 2.1 目标
- 在不引入 PKE 伪标签(可能含噪)的情况下，利用 **True H** 强行拉齐 Moving 侧的 Detector。
- 让 `det_mov` 在此时就开始对齐 `det_fix`。

#### 2.2 具体改动
- **Loss 新增**:
  1.  **Warp Moving Detection**:
      $$P'_{mov} = \text{Warp}(det\_mov, H_{0\to1}^{-1})$$
      将 Moving 预测图 warp 回 Fix 空间。
  2.  **Geometric Consistency Loss**:
      $$l_{geo\_warm} = \text{Dice}(det\_fix, P'_{mov})$$
  3.  **Total Loss**:
      $$L = l_{det\_warm} + l_{des\_warmup} + \lambda_{geo} \cdot l_{geo\_warm}$$
      建议 $\lambda_{geo} \approx 0.5 - 1.0$.

---

### 三、阶段 3：正式 PKE 联合训练 (Phase 3)
**周期**：Epoch 41+
**状态**：PKE **ON**

#### 3.1 目标
- 启用 PKE 自动扩充伪标签，捕捉非血管特征 (如病灶、纹理)。
- 利用前两阶段打下的稳固基础 (Detector 准, Descriptor 强, 几何对齐)，防止 PKE 漂移。

#### 3.2 PKE 策略收紧
- **Content Threshold**: 训练时收紧至 **0.7 - 0.8** (宁缺毋滥)。验证时可保持 0.9。
- **Max New Points**: 限制每张图每轮新增点数 (e.g., max 200-500)，防爆炸。
- **Vessel Mask Filtering**: 仅在血管掩码范围内生成候选点 (已实现)。

#### 3.3 Loss 结构
$$L = l_{det\_pke} + l_{geo\_pke} + l_{des\_gt}$$
- $l_{det\_pke}$: 拟合 PKE 动态标签。
- $l_{geo\_pke}$: PKE 内部的几何一致性。
- **$l_{des\_gt}$ (关键调整)**：即使在 Phase 3，我们依然**坚持使用 GT H (`H_0to1`) + GT 血管分叉点**来计算描述子损失。
  - **原因**：防止 PKE 探索过程中的噪声点污染描述子特征空间。检测器负责“探索”，描述子负责“守住底线”，作为整个训练过程的稳定锚点。

---

## 4. 验证与监控 (Sanity Check)

### 4.1 独立验证通路
- **Fixed/Moving GT Alignment**:
  - 在验证集随机抽取样本，仅使用 `vessel_keypoints` + `H_0to1` 计算匹配误差 (MSE)。
  - 这代表了“完美检测器”下的描述子/几何上限。
- **Keypoint Visualization**:
  - 重点关注 `fixed_kps.png` (绿点) 是否贴合 `vessel_keypoints_extracted.png` (红点)。
### 4.2 可视化监控
- **Real Data**: 重点观察真实数据的配准效果，特别是血管不对齐区域的校正。
- **Gen Data**: 观察生成数据的 `fixed_kps.png` 是否保持稳定，防止灾难性遗忘。

---

## 阶段 4：Sim-to-Real 全量 PKE 微调 (Phase 4)
**周期**：Epoch 100+
**状态**：**Computed H** + **Full PKE**

### 4.1 核心发现与策略
根据 `operation_pre_filtered_octfa_dataset.py`，真实数据集中包含足够数量的配对关键点 ($>4$)，这允许我们**现场计算出可靠的全图单应性矩阵 ($H_{computed}$)**。
这意味着：
1.  我们不需要妥协使用稀疏监督。
2.  我们可以将真实数据视作拥有“准 Ground Truth H”的数据。
3.  我们可以**完全复用 Phase 3 的 PKE 训练流程**，直接在真实图像上跑全量 PKE。

### 4.2 数据流设计 (Hybrid Batch)
1.  **Replay Buffer (Syn, ~20%)**:
    -   $H_{gt}$来自数据集预设。
2.  **Real Data (Real, ~80%)**:
    -   $H_{computed}$ 由 DataLoader 根据 $P_{fix}, P_{mov}$ 实时计算 (`cv2.findHomography` with RANSAC)。
    -   利用 $H_{computed}$ 对 image1 进行预对齐 (pre-alignment)，或者直接作为 $T_{0\to1}$ 传入模型。

### 4.3 训练与 Loss (Hybrid Supervision)
$$L_{real} = L_{det\_pke} + L_{geo\_pke} + L_{des\_total}$$

1.  **PKE Detection & Geometric Loss**:
    -   利用 $H_{computed}$ 在全图范围内挖掘新点，计算 $L_{det}$ 和 $L_{geo}$。
2.  **Hybrid Descriptor Loss ($L_{des\_total}$)** (核心防过拟合策略):
    -   **Anchor Loss**: 在人工标注 GT 点 ($P_{fix}, P_{mov}$) 处计算强监督 Loss。保证“基准不飘”。
    -   **Exploration Loss**: 在 **PKE 挖掘出的所有新点对**上计算自监督 Loss。
        -   利用 $H_{computed}$ 建立对应关系。
        -   即使真实数据少，PKE 也能在每张图上挖掘出成百上千个新特征点，极大丰富训练样本，**彻底解决过拟合问题**。
    -   $$L_{des\_total} = L_{des}(GT) + \lambda \cdot L_{des}(PKE_{mined})$$

