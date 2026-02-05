# SuperRetina v8: CF-Anchor Dual-Path Multimodal Alignment (No PKE)
我现在在做一个课题，目标是基于已有的生成数据集（结构完全相同且对齐的 cf-oct-fa 图像对）训练出一个支持 cf-fa，cf-oct（cf 均为 fix）的多模态配准模型。之前尝试直接预测放射参数或用 fractMorph2D 做细粒度形变，都在大尺度旋转（15°、30°）和大位移场景下效果较差。因此 v8 版本选择**完全取消 PKE 机制**，回到一个更直接、更稳定的“CF 作为锚点、其它模态模仿 CF 特征”的跨模态对齐方案。

> **Overview**: v8 以官方 SuperRetina 为 CF 锚点，只训练 Moving Encoder，使 FA/OCT 特征在精确的 H\_0to1 标定下对齐到 CF 特征空间；整个训练过程中不使用 PKE，而是依靠稠密特征对齐和检测头蒸馏来获得足够多、稳定的跨模态匹配点。

---

## 1. 模型结构 (Architecture)

1. **Dual-Path Encoder（双路编码器）**
   - `encoder_fix`: 从 `weights/SuperRetina.pth` 初始化，**全程冻结**，只作为 CF 锚点分支。
   - `encoder_mov`: 初始化为同结构的新编码器，**仅在 Moving 模态（FA/OCT）上训练**。
2. **Head 设计**
   - 描述子头 (`convDa/Db/Dc + trans_conv`)：两支共享，同一套权重。
   - 检测头 (`dconv_up* + conv_last`)：两支共享，保持与官方 SuperRetina 一致的关键点风格。
3. **推理时使用方式**
   - CF 图像始终走 `encoder_fix`；FA/OCT 图像走 `encoder_mov`；两边经共享 Head 输出 `det_fix/mov` 和 `desc_fix/mov`，再用 NMS + 描述子匹配 + RANSAC 估计放射矩阵。

---

## 2. 训练数据与几何信息

1. **生成数据集 FIVES（cf-oct-fa）**
   - 每对样本包含：`I_cf`（fix）、`I_mov`（fa/oct）、精确的仿射/投影矩阵 `H_0to1`，以及 CF 上的血管分割 `vessel_mask_cf`。
2. **真实验证数据 CFFA / CFOCT 等**
   - 只用于 `validate/test`，配准评估与 `test_on_real` 保持一致；  
   - 训练阶段**不再对真实数据做几何监督**。

---

## 3. 训练目标 (Loss Design, 无 PKE)

### 3.1 描述子稠密对齐 (Dense Descriptor Alignment)

利用精确的 `H_0to1`，把 Moving 特征场 warp 回 CF 坐标系，在**血管区域上做稠密对齐**：

1. 下采样 CF 血管掩膜到特征图尺寸：  
   `mask_feat = downsample(vessel_mask_cf)  # [B,1,H_feat,W_feat]`
2. 提取特征向量：
   - `feat_fix = desc_fix[b].view(C, -1).T   # [N, C]`
   - `feat_mov = desc_mov[b].warp(H_0to1).view(C, -1).T`
3. 只在 `mask_feat>0.5` 的位置做 L2/InfoNCE：
   - 基线版本：`L_desc = ||feat_fix_vessel - feat_mov_vessel||_2^2`  
   - 后续可选：改为 InfoNCE（anchor=CF，positive=warp mov，negative=同图其它像素）。

目标：**让 encoder\_mov 在血管区域学到与 encoder\_fix 相同的结构特征分布。**

### 3.2 检测头蒸馏 (Detector Distillation, Mov→Fix)

1. 只保留 CF 分支原有的检测行为：
   - `det_fix` 不强行拟合任何新 GT，只在需要时做轻微 fine-tune。
2. 对 Moving 分支的检测图采用“warp 后拟合 CF”的蒸馏：
   - `det_mov_warp = warp(det_mov, H_0to1)`  
   - 损失：`L_det_mov = Dice(det_mov_warp, det_fix.detach())`

这样 Moving 分支学会在与 CF 对齐的位置产生相似的关键点分布，但**不会额外引入 PKE 扩点、value_map、对称 mask 等强约束**。

### 3.3 总损失与权重

- `L_total = λ_desc * L_desc + λ_det * L_det_mov`
  - 初始建议：`λ_desc = 1.0, λ_det = 0.5`，优先保证特征空间的跨模态对齐。

---

## 4. 训练流程 (v8)

1. **阶段设置**
   - 不再区分 Phase 0 / Phase 1，整个训练过程使用同一套 dense 对齐 + 检测蒸馏损失。
2. **初始化**
   - 载入官方 `SuperRetina.pth`；  
   - 复制 encoder 权重到 `encoder_fix` / `encoder_mov`，然后**冻结 encoder\_fix、解冻 encoder\_mov**。
3. **优化器**
   - 只对 `encoder_mov` + Heads 更新参数（可以对 Heads 用稍大的 LR）。

---

## 5. 验证与可视化

1. **验证协议**
   - 维持 `validate()` 逻辑：NMS→BFMatcher→RANSAC→MEE/MAE/AUC；  
   - 在 CFFA/CFOCT 上评估跨模态匹配与配准精度。
2. **可视化**
   - 每若干 epoch 保存：
     - `*_matches.png`：关注匹配数量和几何正确性；  
     - `*_desc_pca_fix/mov.png`：检查 CF / FA(OCT) 特征颜色分布是否一致；  
     - `*_checkerboard.png`：观察大尺度旋转/位移下的对齐效果。

---

## 6. 预期效果

1. 在生成数据上，CF 与 FA/OCT 之间的描述子在血管区域内分布基本一致，PCA 可视化颜色分布相近。  
2. 在真实 CFFA/CFOCT 数据上，关键点分布密集且稳定，匹配数量充足，RANSAC 能在大旋转、大位移场景下估计出合理的放射矩阵。  
3. 相比直接预测放射参数或像素级形变，基于显式关键点 + 单应矩阵估计的策略在全局刚体/仿射变换下具有更好的收敛性和可解释性。
