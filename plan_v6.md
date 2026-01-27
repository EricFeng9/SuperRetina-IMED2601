# SuperRetina v6: Phase-Aligned Warmup + Symmetric Hybrid Multimodal Registration

> **Overview**: v6.2 方案专注于解决“模态转换”与“几何鲁棒”两个核心痛点的解耦。通过 **Phase-Aligned Warmup (对齐热身)** 阶段让双路编码器在解剖结构对齐的情况下先学会特征映射，再通过 **Symmetric Mask Constraint (对称掩码约束)** 保证关键点检测的纯净度。

---

## 1. 网络架构 (Network Architecture)

### 1.1 Dual-Path Encoder (双路编码器)
*   **Path A (Fix Encoder)**: 处理 CF 图像（暗线特征）。
*   **Path B (Mov Encoder)**: 处理 FA/OCTA 图像（亮线特征）。
*   **权重**: 独立，不共享。

### 1.2 Shared Heads (共享多任务头)
*   **Detector Head**: 共享，输出关键点 Heatmap。
*   **Descriptor Head**: 共享，输出 L2 归一化的特征向量。

---

## 2. 课程学习策略 (Curriculum Learning)

### 2.0 Phase 0: Modality Alignment Warmup (模态对齐热身)
*   **目标**: 解耦模态差异。让双路 Encoder 在无几何畸变的对齐状态下，学习将 CF 的“黑血管”与 FA 的“白血管”映射到同一描述子分布。
*   **操作**:
    1.  **输入**: 使用原图 `fix` 和未形变的 `moving_origin` (解剖全对齐)。
    2.  **约束**: (已修改) 取消单向 Grident Detach，允许双路联合优化 (Joint Optimization)，利用 InfoNCE 的强对比学习特性防止特征坍缩。
    3.  **Loss: 高效密集 InfoNCE (Efficient Dense InfoNCE)**:
        *   **计算尺度**: 在 $1/8$ 特征图分辨率 ($64 \times 64$) 执行。
        *   **分层采样 (Stratified Sampling)**: 动态选取 $N$ (例如 1024) 个锚点点对。
            *   **50% 强血管样**: 从 `vessel_mask` 区域内选取，确保核心解剖结构的极精对齐。
            *   **50% 全图随机样**: 从全图（含背景）随机选取，保证描述子在非血管区域、视盘、渗出等处的泛化转换能力。
        *   **负样本池**: 同一张 $64 \times 64$ 特征图内的所有像素点作为对比负样本。
*   **关键**: 此阶段 **不训练检测头 (Detector)**，专注于对抗多模态纹理鸿沟，防止后续 PKE 乱匹配。

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

1.  **Phase 0 完成后**: 哪怕不训练检测头，只用随机采样点也能在对齐图上获得极佳的描述子距离。
2.  **Phase 1 开始后**: Heatmap 会迅速响应血管分叉，由于已经有了 Phase 0 的特征对齐功底，匹配连线将非常整齐。
3.  **对称约束生效后**: `moving_kpts.png` 上的亮斑、边缘伪影探测点将完全消失。
