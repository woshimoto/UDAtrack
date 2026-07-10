# SGDP-Track Code Explanation

> **Legacy document.** This file describes the precursor implementation on
> `codex/sgdp-track-improvements`. The paper-aligned DriftGuard implementation
> is documented in `README.md`; its canonical controls are
> `--evidence_topk`, `--evidence_score_thresh`, and
> `--state_update_thresh`.

本文档说明当前分支 `codex/sgdp-track-improvements` 相对原始 UDAtrack / DKGTrack 代码做了哪些改进、改进代码在哪里、关键代码是什么，以及每一部分实现的功能。

当前分支：

```bash
codex/sgdp-track-improvements
```

## 1. 改动文件总览

本次改动涉及 8 个文件：

| 文件 | 作用 | 改进类型 |
| --- | --- | --- |
| `main.py` | 训练 / 推理命令行参数入口 | 新增 SGDP、反事实损失、不确定性损失、语义状态参数 |
| `models/deformable_transformer_plus.py` | Deformable Transformer 主体 | 实现 state-guided dual purification：空间 token pruning、通道净化、query uncertainty、SGDP 辅助输出 |
| `models/transrmot_pro.py` | RMOT 主模型和训练 criterion | 实现 reliability-aware RACL、反事实 distractor loss、不确定性监督、semantic state EMA |
| `configs/dkgtrack_rmot_train.sh` | Refer-KITTI v2 训练脚本 | 接入新损失和 SGDP 参数 |
| `configs/dkgtrack_rmot_train_rk.sh` | Refer-KITTI 训练脚本 | 接入新损失和 SGDP 参数 |
| `configs/dkgtrack_rmot_test.sh` | Refer-KITTI v2 测试脚本 | 接入 SGDP 推理参数 |
| `configs/dkgtrack_rmot_test_rk.sh` | Refer-KITTI 测试脚本 | 接入 SGDP 推理参数 |
| `README.md` | 项目说明 | 补充新方法说明和运行参数 |

## 2. 整体功能目标

本次代码把原来较初步的 SGDP 思路补成更完整的 SGDP-Track 实现，主要包括：

1. Reliability-conditioned semantic state
   - 根据当前帧可靠 track query 动态维护语义状态。
   - 后续帧使用这个状态指导视觉 token 选择，而不只依赖静态文本原型。

2. State-guided dual purification
   - 空间层面：根据 static / motion / semantic state 三类响应对 visual tokens 打分，只保留语义相关 token。
   - 通道层面：用 channel gate 在 static prototype 和 motion prototype 之间做自适应融合。

3. Reliability-aware contrastive learning
   - 使用 IoU 作为可靠性权重。
   - 为同一目标维护跨帧 memory positive。
   - 用其他 query 和其他目标 memory 作为 hard negatives。

4. Counterfactual distractor purification
   - 从高语义响应但不在 referred boxes 内的 tokens 中挖 distractor。
   - 让 query 靠近目标区域 tokens，远离 distractor tokens。

5. Channel uncertainty calibration
   - 从 channel gate 得到 query uncertainty。
   - 用 `1 - IoU^beta` 作为监督目标，使不可靠匹配对应更高 uncertainty。

## 3. `main.py` 参数入口改进

位置：

```text
main.py:142-147
main.py:213-223
```

核心代码：

```python
parser.add_argument('--cf_loss_coef', default=1.0, type=float,
                    help='loss weight for counterfactual distractor purification')
parser.add_argument('--unc_loss_coef', default=0.5, type=float,
                    help='loss weight for channel uncertainty calibration')
parser.add_argument('--cf_score_thresh', default=0.35, type=float,
                    help='semantic score threshold used to mine counterfactual distractor tokens')
```

```python
parser.add_argument('--sgdp_topk', default=300, type=int,
                    help='number of semantic-relevant visual tokens retained by SGDP spatial pruning')
parser.add_argument('--sgdp_k_min', default=64, type=int,
                    help='minimum token budget for adaptive SGDP pruning')
parser.add_argument('--sgdp_adaptive_topk', action='store_true', default=True,
                    help='adapt SGDP token budget from semantic-score entropy')
parser.add_argument('--sgdp_disable_adaptive_topk', dest='sgdp_adaptive_topk', action='store_false',
                    help='use fixed --sgdp_topk instead of entropy-adaptive SGDP budget')
parser.add_argument('--semantic_state_momentum', default=0.8, type=float,
                    help='EMA momentum for reliability-conditioned semantic state')
parser.add_argument('--semantic_state_threshold', default=0.05, type=float,
                    help='minimum reliability required to update the semantic state')
```

实现功能：

- `--cf_loss_coef`：控制反事实 distractor loss 权重。
- `--unc_loss_coef`：控制 uncertainty calibration loss 权重。
- `--cf_score_thresh`：控制哪些 token 被认为是高响应 distractor 候选。
- `--sgdp_topk`：空间净化最多保留多少 visual tokens。
- `--sgdp_k_min`：自适应 Top-K 的最小 token 预算。
- `--sgdp_adaptive_topk`：根据语义响应熵动态决定保留 token 数。
- `--semantic_state_momentum`：语义状态 EMA 更新动量。
- `--semantic_state_threshold`：只有可靠性超过该阈值的 query 才用于更新语义状态。

## 4. Transformer 中的 SGDP 改进

文件：

```text
models/deformable_transformer_plus.py
```

### 4.1 Transformer 构造函数接入 SGDP 参数

位置：

```text
models/deformable_transformer_plus.py:39-69
models/deformable_transformer_plus.py:946-966
```

核心代码：

```python
def __init__(..., extra_track_attn=False,
             sgdp_topk=300, sgdp_k_min=64, sgdp_adaptive_topk=True):
```

```python
decoder_layer = DeformableTransformerDecoderLayer(
    d_model, dim_feedforward,
    dropout, activation,
    num_feature_levels, nhead, dec_n_points, decoder_self_cross,
    sigmoid_attn=sigmoid_attn, extra_track_attn=extra_track_attn,
    sgdp_topk=sgdp_topk,
    sgdp_k_min=sgdp_k_min,
    sgdp_adaptive_topk=sgdp_adaptive_topk,
)
```

```python
return DeformableTransformer(
    ...
    sgdp_topk=args.sgdp_topk,
    sgdp_k_min=args.sgdp_k_min,
    sgdp_adaptive_topk=args.sgdp_adaptive_topk,
)
```

实现功能：

- 将命令行里的 SGDP 参数真正传入 Transformer decoder layer。
- 后续空间 pruning 的 token budget 由这些参数控制。

### 4.2 Transformer forward 支持 semantic state，并返回 SGDP 辅助信息

位置：

```text
models/deformable_transformer_plus.py:154-226
```

核心代码：

```python
def forward(self, srcs, masks, pos_embeds, query_embed=None,
            sentence_embeds=None, text_dict=None, ref_pts=None,
            static_feat=None, motion_feat=None, semantic_state=None):
```

```python
token_centers = self.get_token_centers(spatial_shapes, valid_ratios, memory.device)
hs, inter_references, sgdp_aux = self.decoder(
    tgt, reference_points, memory, spatial_shapes, level_start_index,
    valid_ratios, query_embed, mask_flatten, lvl_pos_embed_flatten,
    sentence_embeds, text_dict,
    static_feat=static_feat,
    motion_feat=motion_feat,
    semantic_state=semantic_state,
    token_centers=token_centers,
)
```

```python
return hs, init_reference_out, inter_references_out, None, None, sgdp_aux
```

实现功能：

- 原始 Transformer 只返回检测需要的 decoder features 和 reference points。
- 现在额外返回 `sgdp_aux`，里面包含：
  - `token_scores`
  - `token_features`
  - `token_centers`
  - `token_keep`
  - `query_uncertainty`
- 这些辅助信息会被后续 `loss_counterfactual` 和 `loss_uncertainty` 使用。

### 4.3 生成 visual token 的归一化中心坐标

位置：

```text
models/deformable_transformer_plus.py:229
```

核心代码：

```python
@staticmethod
def get_token_centers(spatial_shapes, valid_ratios, device):
    centers = []
    bs = valid_ratios.shape[0]
    for lvl, (h, w) in enumerate(spatial_shapes.tolist()):
        ...
    return torch.cat(centers, dim=1)
```

实现功能：

- 为每个多尺度 feature map token 生成 `[x, y]` 中心坐标。
- 后续反事实损失需要判断 token 是否落在目标 box 内。

### 4.4 空间门控与通道门控

位置：

```text
models/deformable_transformer_plus.py:318-366
```

核心代码：

```python
self.spatial_gate = nn.Sequential(
    nn.Linear(d_model * 3, d_model),
    nn.ReLU(),
    nn.Linear(d_model, d_model // 4),
    nn.ReLU(),
    nn.Linear(d_model // 4, 1),
    nn.Sigmoid()
)
self.channel_gate = nn.Sequential(
    nn.Linear(d_model, d_model // 2),
    nn.ReLU(),
    nn.Linear(d_model // 2, d_model),
    nn.Sigmoid()
)
self.beta = nn.Parameter(torch.tensor(0.0))
```

实现功能：

- `spatial_gate` 输入是三个语义响应拼接：
  - `src * static_feat`
  - `src * motion_feat`
  - `src * semantic_state`
- 输出每个视觉 token 的语义相关性分数。
- `channel_gate` 输出每个 query/channel 的静态语义与运动语义融合比例。
- `beta` 是可学习强度，训练初始为 0，避免一开始破坏原模型特征。

### 4.5 语义 Top-K 空间净化

位置：

```text
models/deformable_transformer_plus.py:408-447
```

核心代码：

```python
semantic_response = torch.cat(
    [src * static_feat, src * motion_feat, src * semantic_state],
    dim=-1,
)
spatial_score = self.spatial_gate(semantic_response)
score = spatial_score.squeeze(-1).transpose(0, 1)
```

```python
if self.sgdp_adaptive_topk and k_min < k_max:
    safe_score = score.masked_fill(~valid, 0.0).clamp_min(1e-6)
    prob = safe_score / safe_score.sum(dim=1, keepdim=True).clamp_min(1e-6)
    entropy = -(prob * prob.clamp_min(1e-6).log()).sum(dim=1)
    entropy = entropy / max(math.log(score.shape[-1]), 1e-6)
    budgets = k_min + ((k_max - k_min) * entropy).floor().long()
else:
    budgets = score.new_full((score.shape[0],), k_max, dtype=torch.long)
```

```python
for b, budget in enumerate(budgets.tolist()):
    if budget > 0:
        keep[b].scatter_(0, torch.topk(score[b], budget, dim=0).indices, True)
pruned_src = src * keep
```

实现功能：

- 计算每个视觉 token 与文本静态属性、运动属性、历史语义状态的相关性。
- 根据 score 保留最相关 token，其余 token 置零。
- 当响应分布熵较高时，保留更多 token；响应更集中时，保留更少 token。
- 这样可以降低 distractor token 对 cross-attention 的干扰。

### 4.6 通道净化与 query uncertainty

位置：

```text
models/deformable_transformer_plus.py:449-522
```

核心代码：

```python
alpha = self.channel_gate(tgt2)
lang_refined = alpha * static_feat + (1 - alpha) * motion_feat
tgt = tgt + self.dropout1(tgt2) + self.beta * lang_refined
query_uncertainty = alpha.mean(dim=-1)
```

实现功能：

- `alpha` 越大，query 更偏向 static prototype。
- `1 - alpha` 越大，query 更偏向 motion prototype。
- `query_uncertainty = alpha.mean(dim=-1)` 被作为通道门控不确定性估计，后续由 `loss_uncertainty` 监督。

### 4.7 Decoder 汇总 SGDP 辅助信息

位置：

```text
models/deformable_transformer_plus.py:587-620
models/deformable_transformer_plus.py:656-658
```

核心代码：

```python
sgdp_aux = {}
...
output, layer_sgdp_aux = layer(
    output, query_pos, reference_points_input, src_level,
    src_spatial_shapes, src_level_start_index,
    src_padding_mask, lvl_pos_embed_flatten,
    static_feat=static_feat,
    motion_feat=motion_feat,
    semantic_state=semantic_state,
)
if layer_sgdp_aux:
    sgdp_aux = layer_sgdp_aux
    if token_centers is not None:
        sgdp_aux['token_centers'] = token_centers
```

```python
return torch.stack(intermediate), torch.stack(intermediate_reference_points), sgdp_aux
```

实现功能：

- 每层 decoder 都可以产生 SGDP 信息。
- 当前实现保留最后一层有效 SGDP 信息，传回 `TransRMOT`。

## 5. `TransRMOT` 和训练损失改进

文件：

```text
models/transrmot_pro.py
```

### 5.1 ClipMatcher 新增 RACL memory 和新损失参数

位置：

```text
models/transrmot_pro.py:55-95
```

核心代码：

```python
def __init__(..., racl_beta=2.0, racl_temperature=0.07,
             racl_num_negatives=50, cf_score_thresh=0.35):
    ...
    self.racl_beta = racl_beta
    self.racl_temperature = racl_temperature
    self.racl_num_negatives = racl_num_negatives
    self.cf_score_thresh = cf_score_thresh
    self.racl_memory = {}
```

```python
def initialize_for_single_clip(...):
    ...
    self.racl_memory = {}
```

实现功能：

- 每个 clip 单独维护一个 `racl_memory`。
- `racl_memory[obj_id]` 保存该目标上一些可靠帧的 query embedding，作为后续帧正样本。
- 每个 clip 开始时清空 memory，避免跨视频污染。

### 5.2 Reliability-aware contrastive loss

位置：

```text
models/transrmot_pro.py:229-291
```

核心代码：

```python
pred_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes)
target_xyxy = box_ops.box_cxcywh_to_xyxy(target_boxes)
reliability = torch.diag(box_ops.box_iou(pred_xyxy, target_xyxy)[0]).clamp(0, 1)
reliability = reliability.pow(self.racl_beta).detach()
```

```python
if obj_id in self.racl_memory:
    positive = self.racl_memory[obj_id].to(anchor.device).view(1, -1)
else:
    positive = text_embeds[batch_id].unsqueeze(0)
```

```python
memory_negatives = [
    emb.to(anchor.device).view(1, -1)
    for mem_obj_id, emb in self.racl_memory.items()
    if mem_obj_id != obj_id
]
```

```python
logits = torch.cat([pos_logits, neg_logits], dim=1) / self.racl_temperature
per_anchor_loss = F.cross_entropy(logits, labels, reduction='none')
weight = reliability[row].clamp_min(1e-6)
total_loss = total_loss + per_anchor_loss.sum() * weight
```

```python
for obj_id, embedding in memory_updates:
    self.racl_memory[obj_id] = embedding
```

实现功能：

- 用 IoU 衡量当前匹配是否可靠。
- 可靠匹配对 contrastive learning 贡献更大。
- 同一目标的历史 embedding 作为 positive，比只用文本 positive 更稳定。
- 当前帧其他 queries 和其他目标 memory 作为 negatives，提高跨目标区分能力。

### 5.3 反事实 distractor loss

位置：

```text
models/transrmot_pro.py:307-378
```

核心代码：

```python
required = ('sgdp_token_scores', 'sgdp_token_features', 'sgdp_token_centers', 'query_feats')
if any(key not in outputs for key in required):
    return {'loss_counterfactual': outputs['pred_logits'].new_tensor(0.0)}
```

```python
high_response = scores > self.cf_score_thresh
inside_referred = self._points_in_boxes(centers, referred_boxes)
distractor_mask = high_response & (~inside_referred)
distractors = token_features[batch_id, distractor_mask]
```

```python
positive_mask = high_response & self._points_in_boxes(centers, target_box)
positives = token_features[batch_id, positive_mask]
...
positive = F.normalize(positives.mean(dim=0, keepdim=True), p=2, dim=-1)
anchor = F.normalize(query_feats[batch_id, src_idx[row:row + 1]], p=2, dim=-1)
pos_logits = anchor @ positive.t()
neg_logits = anchor @ distractors.t()
logits = torch.cat([pos_logits, neg_logits], dim=1) / self.racl_temperature
```

实现功能：

- 从 SGDP 空间打分中找出高响应 token。
- 如果高响应 token 不在 referred target box 内，就认为它可能是 distractor。
- 对每个正匹配 query：
  - positive 是目标 box 内的高响应 token 平均特征。
  - negative 是目标外高响应 distractor token。
- 用 InfoNCE 形式让 query 远离 distractor，靠近目标区域视觉 token。

### 5.4 判断 token 是否落在 box 内

位置：

```text
models/transrmot_pro.py:296-304
```

核心代码：

```python
@staticmethod
def _points_in_boxes(points, boxes):
    if boxes.numel() == 0 or points.numel() == 0:
        return torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
    px = points[:, 0:1]
    py = points[:, 1:2]
    inside = (px >= xyxy[:, 0]) & (px <= xyxy[:, 2]) & (py >= xyxy[:, 1]) & (py <= xyxy[:, 3])
    return inside.any(dim=1)
```

实现功能：

- 输入 token 中心点和 box，返回每个 token 是否在任一 box 内。
- 是反事实 distractor mining 的基础函数。

### 5.5 Uncertainty calibration loss

位置：

```text
models/transrmot_pro.py:380-405
```

核心代码：

```python
uncertainty = outputs['query_uncertainty']
...
reliability = torch.diag(box_ops.box_iou(pred_xyxy, target_xyxy)[0]).clamp(0, 1).pow(self.racl_beta).detach()
target_uncertainty = 1.0 - reliability
total_loss = total_loss + F.l1_loss(
    uncertainty[batch_id, src_idx],
    target_uncertainty,
    reduction='sum',
)
```

实现功能：

- IoU 越高，说明预测越可靠，目标 uncertainty 越低。
- IoU 越低，说明预测越不可靠，目标 uncertainty 越高。
- 让 channel gate 输出的不确定性与定位可靠性对齐。

### 5.6 Semantic state 维护与更新

位置：

```text
models/transrmot_pro.py:871-907
```

核心代码：

```python
def _get_semantic_state(self, fallback):
    if self._semantic_state is None:
        return fallback
    return self._semantic_state.to(fallback.device, fallback.dtype)
```

```python
reliability = scores.clamp(0, 1) * refers.clamp(0, 1) * localization * (1.0 - uncertainty.clamp(0, 1))
reliability = reliability * active.float()
keep = reliability > self.semantic_state_threshold
```

```python
weights = reliability[keep].view(-1, 1)
state = (query_embed[keep] * weights).sum(dim=0, keepdim=True) / weights.sum().clamp_min(1e-6)
state = F.normalize(state, p=2, dim=-1).view(1, 1, -1)
```

```python
self._semantic_state = F.normalize(
    self.semantic_state_momentum * prev + (1.0 - self.semantic_state_momentum) * state,
    p=2,
    dim=-1,
).detach()
```

实现功能：

- 如果当前还没有 semantic state，就使用当前文本 static prototype 作为 fallback。
- 根据 track score、refer score、localization IoU、query uncertainty 综合得到 reliability。
- 只用可靠 query embedding 更新 semantic state。
- 用 EMA 方式平滑更新，避免单帧噪声导致状态剧烈变化。

### 5.7 单帧 forward 中生成 prototypes 并传给 Transformer

位置：

```text
models/transrmot_pro.py:957-987
models/transrmot_pro.py:1065-1071
```

核心代码：

```python
if static_feat.size(0) > 0:
    static_prototype = static_feat.mean(dim=0).view(1, 1, -1)
else:
    static_prototype = text_sentence_features.mean(dim=1).view(1, 1, -1)

if motion_feat.size(0) > 0:
    motion_prototype = motion_feat.mean(dim=0).view(1, 1, -1)
else:
    motion_prototype = torch.zeros_like(static_prototype)
```

```python
if semantic_state_override is None:
    semantic_state = self._get_semantic_state(static_prototype)
else:
    semantic_state = semantic_state_override.to(static_prototype.device, static_prototype.dtype)
```

```python
hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, sgdp_aux = \
    self.transformer(
        srcs, masks, pos, track_instances.query_pos,
        text_sentence_features, text_dict,
        ref_pts=track_instances.ref_pts,
        static_feat=static_prototype,
        motion_feat=motion_prototype,
        semantic_state=semantic_state,
    )
```

实现功能：

- 从文本解耦得到 static tokens 和 motion tokens。
- 分别聚合成 static prototype 和 motion prototype。
- 加入历史 semantic state。
- 一起传给 Transformer，实现 state-guided SGDP。

### 5.8 输出 SGDP 辅助信息供 loss 使用

位置：

```text
models/transrmot_pro.py:1115-1125
```

核心代码：

```python
if sgdp_aux:
    if 'token_scores' in sgdp_aux:
        out['sgdp_token_scores'] = sgdp_aux['token_scores']
    if 'token_features' in sgdp_aux:
        out['sgdp_token_features'] = sgdp_aux['token_features']
    if 'token_centers' in sgdp_aux:
        out['sgdp_token_centers'] = sgdp_aux['token_centers']
    if 'token_keep' in sgdp_aux:
        out['sgdp_token_keep'] = sgdp_aux['token_keep']
    if 'query_uncertainty' in sgdp_aux:
        out['query_uncertainty'] = sgdp_aux['query_uncertainty']
```

实现功能：

- 将 Transformer 内部 SGDP 信息写入 `outputs`。
- `loss_counterfactual` 读取 token score / token feature / token center。
- `loss_uncertainty` 读取 query uncertainty。

### 5.9 缓存 query uncertainty，用于后续 semantic state 更新

位置：

```text
models/transrmot_pro.py:1208-1237
```

核心代码：

```python
query_feats = out['query_feats']
...
track_instances.output_embedding = query_feats[0].clone()
...
if 'query_uncertainty' in out:
    track_instances.cache_query_uncertainty = out['query_uncertainty'][0].detach().clone()
else:
    track_instances.cache_query_uncertainty = torch.zeros_like(track_scores)
```

```python
track_instances.query_uncertainty = track_instances.cache_query_uncertainty
```

实现功能：

- 原代码这里会 `pop('query_feats')`，导致后续 loss 不方便继续使用 query features。
- 现在改成读取但不删除。
- 缓存 `query_uncertainty`，让 semantic state 更新时可以降低不确定 query 的影响。

### 5.10 build 阶段接入新 loss

位置：

```text
models/transrmot_pro.py:1465-1520
```

核心代码：

```python
weight_dict.update({
    ...
    'frame_{}_loss_contrastive'.format(i): args.racl_loss_coef,
    'frame_{}_loss_counterfactual'.format(i): args.cf_loss_coef,
    'frame_{}_loss_uncertainty'.format(i): args.unc_loss_coef,
    ...
})
```

```python
losses = ['labels', 'boxes', 'refers', 'loss_contrastive',
          'loss_counterfactual', 'loss_uncertainty']
criterion = ClipMatcher(
    ...
    racl_beta=args.racl_beta,
    racl_temperature=args.racl_temperature,
    racl_num_negatives=args.racl_num_negatives,
    cf_score_thresh=args.cf_score_thresh,
)
```

```python
model = TransRMOT(
    ...
    semantic_state_momentum=args.semantic_state_momentum,
    semantic_state_threshold=args.semantic_state_threshold,
)
```

实现功能：

- 新增 loss 被纳入总 loss。
- 新参数从 `main.py` 传入 criterion 和 model。
- 训练时可以直接通过 shell 脚本或命令行调节权重。

## 6. 配置脚本改进

文件：

```text
configs/dkgtrack_rmot_train.sh
configs/dkgtrack_rmot_train_rk.sh
configs/dkgtrack_rmot_test.sh
configs/dkgtrack_rmot_test_rk.sh
```

训练脚本新增：

```bash
--racl_loss_coef 1 \
--racl_beta 2 \
--racl_temperature 0.07 \
--racl_num_negatives 50 \
--cf_loss_coef 1 \
--unc_loss_coef 0.5 \
--cf_score_thresh 0.35 \
--sgdp_topk 300 \
--sgdp_k_min 64 \
```

测试脚本新增：

```bash
--sgdp_topk 300 \
--sgdp_k_min 64 \
```

实现功能：

- 训练时启用 RACL、counterfactual loss、uncertainty loss。
- 训练和测试都保持相同的 SGDP token selection 设置。
- 路径仍建议通过环境变量传入，例如：

```bash
DATA_ROOT=/path/to/refer-kitti-v2 \
TRAIN_SPLIT=/path/to/refer-kitti-v2.train \
PRETRAIN=/path/to/pretrain.pth \
TEXT_ENCODER_PATH=/path/to/roberta_base \
bash configs/dkgtrack_rmot_train.sh
```

## 7. 消融实验开关

为了配合新的 method 叙事，代码现在提供以下消融开关。默认不传这些参数时，运行完整模型。

| 开关 | 作用 | 对应验证点 |
| --- | --- | --- |
| `--disable_semantic_state` | 不使用跨帧 semantic state，退化为当前文本 static prototype | 验证 language-conditioned semantic state |
| `--disable_state_update` | 不更新 semantic state | 验证 state-safe temporal update |
| `--disable_evidence_pruning` | cross-attention 保留全部视觉 evidence，但仍计算 evidence score | 验证 drift-aware evidence pruning |
| `--disable_channel_rectification` | 关闭 static-motion channel rectification 和 uncertainty 输出 | 验证 uncertainty-gated query rectification |
| `--cf_loss_coef 0` | 关闭 counterfactual evidence disambiguation loss | 验证 counterfactual distractor suppression |
| `--unc_loss_coef 0` | 关闭 uncertainty calibration loss | 验证 uncertainty supervision |
| `--racl_loss_coef 0` | 关闭辅助 identity separation loss | 验证身份分离辅助项 |

推荐的核心消融命令形式：

```bash
DATA_ROOT=/path/to/refer-kitti-v2 \
TRAIN_SPLIT=/path/to/refer-kitti-v2.train \
PRETRAIN=/path/to/pretrain.pth \
TEXT_ENCODER_PATH=/path/to/roberta_base \
bash configs/dkgtrack_rmot_train.sh \
  --disable_semantic_state
```

四个 train/test shell 脚本都支持在命令末尾追加参数，因此同一个脚本可以直接用于完整模型和不同消融设置。

## 8. 数据流说明

### 7.1 前向传播数据流

```text
image + text
  -> backbone extracts multi-scale visual features
  -> text encoder extracts word / sentence features
  -> text decoupling gets static tokens and motion tokens
  -> static_prototype / motion_prototype / semantic_state
  -> DeformableTransformer
      -> spatial_gate scores visual tokens
      -> adaptive Top-K prunes visual tokens
      -> cross-attention on purified visual tokens
      -> channel_gate fuses static / motion prototypes
      -> output query_uncertainty and token-level SGDP aux
  -> TransRMOT detection heads
  -> losses use boxes, refers, contrastive embeddings, SGDP aux
```

### 7.2 训练损失数据流

```text
outputs
  -> loss labels / boxes / refers
  -> loss_contrastive
       uses contrastive_visual, contrastive_text, obj_ids, IoU reliability, racl_memory
  -> loss_counterfactual
       uses sgdp_token_scores, sgdp_token_features, sgdp_token_centers, query_feats
  -> loss_uncertainty
       uses query_uncertainty and IoU reliability
```

### 7.3 Semantic state 更新数据流

```text
track_instances
  -> output_embedding
  -> cache_scores
  -> cache_pred_refers
  -> iou during training
  -> cache_query_uncertainty
  -> reliability
  -> weighted query embedding average
  -> EMA update _semantic_state
```

## 9. 每个改进对应的论文含义

| 代码模块 | 论文中的作用 | 解决的问题 |
| --- | --- | --- |
| `spatial_gate` + `_semantic_topk_prune` | State-guided spatial purification | 降低背景和非目标 token 干扰 |
| `channel_gate` | Channel-wise purification | 在静态属性和运动属性之间自适应选择 |
| `semantic_state` | Reliability-conditioned temporal semantic memory | 避免只依赖当前文本，利用跨帧可靠 query 表征 |
| `loss_contrastive` + `racl_memory` | Reliability-aware association learning | 增强同一目标跨帧一致性，压开不同目标 |
| `loss_counterfactual` | Counterfactual distractor suppression | 显式惩罚高响应但非目标区域的干扰 token |
| `loss_uncertainty` | Uncertainty calibration | 让模型知道哪些 query / 通道融合结果不可靠 |

## 10. 服务器上如何定位这些代码

在服务器仓库目录执行：

```bash
git branch --show-current
git rev-parse --short HEAD
```

应看到：

```text
codex/sgdp-track-improvements
7add6e4
```

快速定位所有新增关键词：

```bash
git grep -n "loss_counterfactual\\|loss_uncertainty\\|semantic_state\\|sgdp_aux\\|sgdp_k_min\\|racl_memory\\|query_uncertainty"
```

查看关键文件：

```bash
sed -n '135,225p' main.py
sed -n '318,522p' models/deformable_transformer_plus.py
sed -n '229,405p' models/transrmot_pro.py
sed -n '871,1125p' models/transrmot_pro.py
sed -n '1440,1525p' models/transrmot_pro.py
```

## 11. 训练前建议验证

由于当前本地机器没有 PyTorch，完整 forward 和训练启动需要在服务器验证。建议按下面顺序执行：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

```bash
python -m py_compile main.py models/transrmot_pro.py models/deformable_transformer_plus.py
```

```bash
cd models/ops
bash make.sh
python test.py
cd ../..
```

然后启动训练：

```bash
DATA_ROOT=/path/to/refer-kitti-v2 \
TRAIN_SPLIT=/path/to/refer-kitti-v2.train \
PRETRAIN=/path/to/pretrain.pth \
TEXT_ENCODER_PATH=/path/to/roberta_base \
bash configs/dkgtrack_rmot_train.sh
```

如果使用 Refer-KITTI 原版：

```bash
DATA_ROOT=/path/to/Refer-KITTI \
TRAIN_SPLIT=/path/to/refer-kitti.train \
PRETRAIN=/path/to/pretrain.pth \
TEXT_ENCODER_PATH=/path/to/roberta_base \
bash configs/dkgtrack_rmot_train_rk.sh
```

## 12. 可能需要重点观察的训练日志

训练时建议重点确认 loss 字段是否出现：

```text
loss_contrastive
loss_counterfactual
loss_uncertainty
```

如果 `loss_counterfactual` 长期为 0，常见原因是：

- `sgdp_token_scores` 没有正确传出。
- `cf_score_thresh` 太高，挖不到高响应 distractor token。
- 当前 batch 中 referred boxes 太少或匹配为空。

如果 `loss_uncertainty` 长期为 0，常见原因是：

- `query_uncertainty` 没有从 Transformer 传回。
- 匹配为空，即 `indices` 中没有有效 `src_idx / tgt_idx`。

如果训练显存明显增加，可以优先调小：

```bash
--sgdp_topk 200
--racl_num_negatives 20
```

## 13. 小结

本次改进不是简单增加几个 loss，而是把视觉 token 选择、文本静态/运动原型、跨帧语义状态、query uncertainty 和可靠性加权训练串成了一条完整路径：

```text
semantic prototypes + semantic state
  -> SGDP spatial/channel purification
  -> SGDP auxiliary outputs
  -> counterfactual / uncertainty / reliability-aware contrastive losses
  -> reliable query embeddings update semantic state
```

因此它对应的是论文中“where to look”和“what to trust”两个核心问题：

- where to look：通过 SGDP 空间净化和反事实 distractor loss 找到真正相关的视觉 token。
- what to trust：通过 reliability-aware RACL、uncertainty calibration 和 semantic state 只信任可靠 query。
