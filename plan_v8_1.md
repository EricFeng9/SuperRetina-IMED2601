# SuperRetina v8.1: CF-Anchor Alignment + Gentle PKE Refinement
我现在在做一个课题，目标是基于已有的生成数据集（结构完全相同且对齐的 cf-oct-fa 图像对）训练出一个支持 cf-fa，cf-oct（cf 均为 fix）的多模态配准模型。v8.1 采用“CF 锚点 + 双路编码器”的结构，并在稠密特征对齐的基础上**引入一种温和的 PKE 机制**：不通过 PKE 改写监督标签或大范围抑制响应，而是将 PKE 选出的稳定关键点视作高权重区域，用来加权描述子和检测损失，在不破坏全局结构的前提下强化高质量匹配点。

> **Overview**: v8.1 由两个阶段组成：  
> 1）阶段一：利用稠密描述子对齐和检测蒸馏，使 FA/OCT 特征在精确 H\_0to1 标定下对齐到 CF 特征空间；  
> 2）阶段二：启用温和版 PKE，根据动态 value\_map 构建 per-pixel 权重图，对损失进行加权，从而强调长期稳定的关键点区域。

---

## 1. 模型结构

1. **Dual-Path Encoder**
   - `encoder_fix`: 从官方 SuperRetina 初始化，全程冻结，作为 CF 锚点。
   - `encoder_mov`: 可训练，只在 FA/OCT 上更新。
2. **共享 Head**
   - 描述子头和检测头与 v8 完全一致，推理时统一用 NMS + BFMatcher + RANSAC 求放射矩阵。
3. **两个训练阶段**
   - 阶段一：仅使用稠密描述子对齐和 Moving 检测头蒸馏，不启用 PKE；  
   - 阶段二：在阶段一收敛的基础上启用温和版 PKE，对损失进行加权精炼。

---

## 2. 阶段二：温和版 PKE

### 2.1 PKE 的定位变化

1. **不用 PKE 生成“硬标签”去完全重写检测监督**  
   - 不再用 `enhanced_label` 直接替换 CF 分支的监督目标；  
   - 也不再用 PKE 驱动对 Mov 分支的 Dice 监督，只做轻量的蒸馏。
2. **不再让 value\_map 对所有未出现位置做强衰减**  
   - value\_map 只用来标记“长期稳定出现的好点”，  
   - 不当作“全局遗忘机制”。
3. **PKE 的主要作用：提供一个 per-pixel 的权重图，放大稳定关键点在损失中的贡献，而不是屏蔽其他点。**

### 2.2 关键点候选与过滤（沿用现有 PKE 但放宽）

1. **候选点来源**
   - 在 CF 分支的 `det_fix` 上做 NMS 提取候选关键点；  
   - 可选：对 NMS 阈值和 `nms_size` 做更宽松的配置，保证候选数量充足。
2. **血管 mask 与几何过滤**
   - 仍然只在 CF 血管掩膜上选点，但阈值放宽（如 `mask > 0.0`）；  
   - `geometric_thresh` 下调（例如 0.3–0.4），允许 Mov 检测图对这些点的响应适度波动。
3. **内容过滤（content_thresh）**
   - 保留 Lowe Ratio 策略，但只作为“置信度打分”的一部分：  
   - 通过内容过滤的点给更高权重，未通过的点不直接被删，只是权重较低。

### 2.3 value\_map 的重定义：从“硬选点”到“权重图”

1. **记录机制**
   - 仍然用 `update_value_map` 将多轮 PKE 认为好的点累积到 value\_map 中；  
   - 但将衰减系数设为非常小，或只对高值区域内部做相对衰减，避免“整个图一刀切”变成 0。
2. **构建权重图**
   - 对最终的 value\_map 做归一化：  
     `w = normalize(value_map)`，范围约在 `[1, 1 + α]`。  
   - 在损失中用作 multiplicative weight，而不是 hard mask。

---

## 3. v8.1 阶段的损失设计

在 v8 的 `L_desc` 和 `L_det_mov` 基础上叠加 PKE 权重：

1. **加权描述子对齐 (PKE-Weighted Dense Alignment)**
   - 对 desc 对齐损失加上 PKE 权重：  
     `L_desc_v8_1 = mean( w * ||desc_fix_vessel - desc_mov_vessel||_2^2 )`，  
     其中 `w` 来自 value\_map warp 后的权重图（在特征图尺度上插值）。
   - 作用：**在保持全局对齐的前提下，对长期稳定的关键点区域施加更大惩罚，强化这些位置的跨模态一致性。**

2. **加权检测蒸馏 (PKE-Weighted Detector Distillation)**
   - 基于 v8 的 `L_det_mov = Dice(det_mov_warp, det_fix.detach())`，  
   - 将 PKE 权重作为 soft mask：  
     - 可以采用 weighted Dice，或在输入侧用 `det_mov_warp * (1 + β*w)`。

3. **不再使用对称 mask 作为强抑制项**
   - 放弃 v7 里 Fix/Mov 双边的 `loss_suppress`；  
   - 只保留轻量的背景抑制（如在 CF 分支用掩膜抑制明显噪声），且权重较小。

4. **总损失**
   - `L_total_v8_1 = λ_desc * L_desc_v8_1 + λ_det * L_det_mov_v8_1`  
   - 其中 λ 可以沿用 v8 初始值，之后根据验证效果微调。

---

## 4. 训练流程

1. **阶段一：无 PKE 的基础训练**
   - 使用双路编码器结构，执行稠密描述子对齐和 Moving 检测头蒸馏；  
   - 在生成数据上观察 AUC / MEE 和 matches 分布，达到稳定后进入阶段二。
2. **阶段二：温和 PKE 精炼**
   - 解冻 value\_map / PKE 相关模块；  
   - 启用 PKE 计算权重图，并在损失中使用加权版本；  
   - 注意：**不改变 NMS/匹配/验证逻辑，PKE 只影响训练损失的关注区域和权重分布。**

---

## 5. 评估指标与预期表现

1. 在生成数据上，阶段二相较阶段一能进一步降低血管区域的描述子对齐误差，高权重关键点附近的特征更加一致。  
2. 在真实 CFFA/CFOCT 数据上，关键点分布和 matches 数量保持充足的同时，更集中在血管分叉和主干区域，RANSAC 求得的放射矩阵更加稳定。  
3. value\_map 所对应的权重图类似于可解释的 attention map，有助于分析模型在何处最有信心完成跨模态对齐。

---

## 6. 实验与可视化安排

1. **阶段性可视化**
   - Stage1 末尾、Stage2 中期与末尾，各导出一轮：  
     - `*_matches.png`、`*_checkerboard.png`、`*_desc_pca_*`。  
   - 对比关键点分布、匹配数量和配准效果。
2. **超参扫描（小范围）**
   - 针对 v8.1 单独扫：`geometric_thresh`、`content_thresh`、PKE 权重缩放系数 α/β，  
   - 优先观察：是否会再次出现“关键点过少”的迹象，一旦出现则回退为更温和的设置。
