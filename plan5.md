# SuperRetina v5: Vessel-Guided Multimodal Registration
> **核心哲学**: 融合先验知识 (Vessel Prior) 与自监督学习 (Self-Supervised Learning)，通过课程学习 (Curriculum Learning) 实现从"强引导"到"自由探索"的平滑过渡。

---

## 1. 架构设计 (Architecture)

### 1.1 骨干网络 (Backbone)
*   **Siamese Network (孪生网络)**: 采用完全共享权重的编码器 (Shared Encoder)。
    *   **理由**: 相比 Dual-Path，共享编码器能强制将不同模态映射到同一特征空间，保证描述子的一致性。
    *   **适用性**: 配合输入端的"模态对齐"预处理，Siamese 架构在眼底多模态任务上表现出更强的鲁棒性。

### 1.2 输入预处理 (Input Alignment)
*   **Fix Image (通常为 CF)**: 
    *   不做处理直接输入 (若为黑血管)。
    *   **关键操作**: 如果是 CF 模态（通常通过反色对齐 FA/OCTA），执行 `Img = 1.0 - Img`。
    *   **目的**: 消除“黑血管 vs 白血管”的根本性视觉冲突，使底层边缘特征对齐。
*   **Moving Image (FA/OCTA)**: 保持原样（亮血管）。

---

## 2. 训练策略 (Training Strategy)

### 2.1 三阶段课程学习 (Curriculum Schedule)
我们设计了一个随时间动态调整的 Loss 权重方案，逐步放松对模型的约束。

### 2.1 二阶段训练策略 (Two-Stage Strategy)
修正后的逻辑：在 Phase 1 (GT注入) 不需要特殊加权（全是血管点）。在 Phase 2 (PKE开启) 需要高权重引导模型聚焦血管，抑制背景噪声。

| 阶段 | 周期 (Epochs) | 任务 | Vessel Weight ($W_{vessel}$) | PKE 模式 | 详细逻辑 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1: Warmup** | **1 - 20** | **GT 强监督** | **1.0** (Standard) | **Supervised** | 1. **GT 注入**: 强制使用 GT 分叉点。<br>2. **权重**: 正常权重。因为此时所有样本都是 Positive (血管)，加权无意义。<br>3. **目的**: 快速拉齐特征空间。 |
| **Phase 2: Refinement** | **21 - 500** | **掩码引导挖掘** | **5.0** (High Constant) | **Self-Supervised** | 1. **PKE 开启**: 模型开始在全图搜索关键点（含背景噪声）。<br>2. **高权重压制**: 对落在 Mask 内的点给与 **5倍权重**。迫使模型优先优化血管上的特征，从而间接抑制背景区域的响应。<br>3. **目的**: 在自监督过程中引入先验，消除背景误匹配。 |

### 2.2 损失函数设计 (Loss Functions)

#### A. Vessel-Guided Descriptor Loss (核心创新)
利用血管掩码 (Mask) 对 InfoNCE/Triplet Loss 进行**非对称加权**：
$$ L_{desc} = \frac{1}{N} \sum_{i=1}^{N} w_i \cdot L_{triplet}(p_i, p_i') $$
其中权重 $w_i$ 定义为：
$$ w_i = \begin{cases} W_{vessel} (E) & \text{if } p_i \in \text{Vessel Mask} \\ 1.0 & \text{if } p_i \in \text{Background} \end{cases} $$
*   **$W_{vessel}(E)$**: 随 Epoch $E$ 变化的动态权重（见课程表）。
*   **效果**: 在初期严厉惩罚血管上的误匹配，后期允许模型关注全局一致性。

#### B. Detector Loss (Supervised PKE)
*   **Phase 1**: 使用 Dice Loss 强拟合 GT Heatmap (由 GT 分叉点生成的高斯图)。
*   **Phase 2/3**: 使用 PKE 生成的 `enhanced_label` (基于几何一致性筛选后的点) 作为伪标签。

---

## 3. 数据增强 (Augmentation)
鉴于 CNN 对大角度旋转的不变性较差，我们采取**"稳健优先"**的增强策略：
*   **Detection (找点)**: 允许 `[-30, 30]` 度旋转。Detector 只需要识别"这是个分叉"，对旋转不敏感。
*   **Description (描述)**: 限制在 **`[-10, 10]`** 度旋转。
    *   **理由**: 强迫 CNN 学习 30 度旋转会导致特征坍塌（区分度降低）。
    *   **对策**: 保证小角度下的高精度匹配，大角度旋转依靠 RANSAC 的鲁棒性解决。

---

## 4. 关键超参数
*   `shared_encoder`: **True**
*   `pke_supervised`: Epoch 1-20 True, else False
*   `vessel_weight_max`: 10.0
*   `vessel_weight_min`: 1.0
*   `decay_start_epoch`: 20
*   `decay_end_epoch`: 50

---

## 5. 预期改进
1.  **误匹配 (Mismatches)**: 背景区域的红线将显著减少 (得益于 Vessel-Guided Loss)。
2.  **取点精度 (Keypoint Accuracy)**: 关键点将更精准地落在血管中心 (得益于 Supervised PKE)。
3.  **匹配召回率 (Recall)**: 在小角度 (`<15°`) 场景下，匹配数量和质量将达到最优。