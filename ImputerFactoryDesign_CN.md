# ImageImputer — 模块实现状态

> 最后更新：2026-05-20（已经废弃，以英文文档为准）

## 实现进度总表

| 模块 | 组件 | 状态 | 备注 |
|---|---|---|---|
| **数据类型** | `ImputerConfig` | ✅ 完成 | 共享只读配置：模型元数据 + 加速器 + segmenter_kwargs |
| | `SpatialLayout` | ✅ 完成 | Player↔像素/token 映射元数据 |
| | `PhysicalMask` | ✅ 完成 | 具体掩码：`image_binary_mask` (N,C,H,W) + `text_attention_mask` (N,L) |
| | `ProcessorOutput` | ✅ 完成 | 标准化 HuggingFace 输入封装 |
| **Segmenter** | `BaseSegmenter` | ✅ 完成 | 抽象接口：`get_layout()` + `generate_masks()` |
| | `PatchSegmenter` | ✅ 完成 | 刚性网格，支持 CLIP/SigLIP/SigLIP2 文本掩码 |
| | `SLICSegmenter` | ❌ 超出范围 | CNN 专用；CLIP 聚焦阶段排除 |
| | `GradientGuidedSegmenter` | ⏳ 占位 | 需要梯度提取 + 分水岭布局 |
| | `AdaptiveSegmenter` | ⏳ 占位 | 需要粗到细的细分逻辑 |
| | `HybridSegmenter` | ❌ 超出范围 | 当前阶段不规划 |
| **Masker** | `BaseMasker` | ✅ 完成 | 抽象接口：`apply(ProcessorOutput, PhysicalMask)` |
| | `CrossModalMeanMasker` | ✅ 完成 | 乘性二进制掩码（图像）+ attention_mask 替换（文本） |
| | `AttentionMasker` | ⏳ 占位 | 需要负无穷自注意力注入 |
| **核心** | `ImageImputer` | ✅ 完成 | `forward_1d` + `forward_crossmodal`，含批处理与设备管理 |
| **工厂** | `ImageImputerFactory` | ✅ 完成 | 自动检测模型类型，装配 PatchSegmenter + CrossModalMeanMasker |
| **适配器** | `TensorOps` / `TorchOps` / `JaxOps` | ⏳ 占位 | 接口已定义，实现待完成 |
| **集成** | `VisionLanguageGame` | ✅ 完成 | 薄适配器：委托给 Imputer，约 75 行 |

### 图例
- ✅ 完成 — 已完整实现并通过测试
- ⏳ 占位 — 骨架存在，逻辑待完成
- ❌ 未开始 — 尚未创建

---

## 模块职责与边界

> 每个模块有单一、明确的职责。  
> 各组不得在未与 PM 讨论的情况下跨越这些边界。

### Segmenter（分割器）
**唯一职责**：空间划分 — 定义哪些像素/token 属于哪个玩家。

- **拥有**：布局元数据（`SpatialLayout`）、将 coalition 转换为像素/token 掩码的算法
- **不拥有**：如何施加遮挡（那是 Masker 的职责）、模型前向传播、批处理逻辑
- **接口约定**：
  - `get_layout()` → 返回 `SpatialLayout`（仅调用一次）
  - `generate_masks(coalitions_image, coalitions_text)` → 返回 `PhysicalMask`（每批调用）
- **禁止**：访问 `model`、调用 `processor`、在 `generate_masks` 内操作 GPU 张量
- **允许变化**：网格形状、玩家数量、玩家到像素的映射、掩码生成算法
- **示例**：`PatchSegmenter`（刚性网格）、`GradientGuidedSegmenter`（基于梯度）、`AdaptiveSegmenter`（分数驱动）

### Masker（掩码器）
**唯一职责**：特征遮挡 — 将 `PhysicalMask` 施加到 `ProcessorOutput` 上。

- **拥有**：遮挡策略（乘性、注意力注入等）
- **不拥有**：掩码生成、模型前向传播、批处理逻辑
- **接口约定**：
  - `apply(processor_output, physical_mask)` → 返回 `ProcessorOutput`（已修改）
- **禁止**：访问 `model`、调用 `processor`、从 coalition 生成掩码
- **必须**：修改前克隆输入（绝不改变原始输入）
- **示例**：`CrossModalMeanMasker`（像素乘法 + 注意力替换）、`AttentionMasker`（负无穷注入）

### ImageImputer（图像填充器）
**唯一职责**：编排 — 协调 Segmenter → Masker → 模型前向传播。

- **拥有**：模型、处理器、原始输入（`input_image`/`input_text`）、预处理的 `inputs_original`、批处理循环、设备管理
- **不拥有**：分割算法、遮挡策略
- **接口约定**：
  - `forward_1d(coalitions, batch_size)` → `np.array`（对角线 logits）
  - `forward_crossmodal(coalitions_img, coalitions_txt, batch_size)` → `np.array`（完整矩阵）
- **编排流程**：`Segmenter.generate_masks → Masker.apply → model(**inputs) → 提取 logits`
- **必须**：处理设备放置、批次迭代、边缘情况（txt_bs ≠ img_bs）
- **禁止**：直接实现掩码生成逻辑、无必要地调用 `processor`

### ImageImputerFactory（工厂）
**唯一职责**：装配 — 检查模型并连接正确的组件。

- **拥有**：组件选择逻辑、模型内省、一次性预处理
- **不拥有**：分割、遮挡、前向传播、批处理
- **接口约定**：
  - `build(model, processor, input_image, input_text, accelerator=...)` → `ImageImputer`
- **必须**：推断模型类型、计算文本玩家数、创建适当的 Segmenter+Masker
- **禁止**：运行模型前向传播、生成掩码、施加遮挡

### VisionLanguageGame（位于 `Game/`）
**唯一职责**：shapiq 适配器 — 提供 shapiq 近似器所需的 `value_function` 接口。

- **拥有**：归一化值（`empty_value`/`full_value`）、batch_size、向后兼容的委托属性
- **不拥有**：模型、处理器、掩码、批处理
- **接口约定**：
  - `value_function(coalitions)` → 委托给 `imputer.forward_1d`
  - `value_function_crossmodal(img, txt)` → 委托给 `imputer.forward_crossmodal`
- **禁止**：导入 `torch`、直接调用 `model`、生成掩码、实现 value_function 逻辑

### 数据类型（位于 `ImputerFactory/data.py`）
**唯一职责**：通用数据协议 — 定义模块间传递对象的形状和语义。

- `ImputerConfig`：共享只读配置（由 Factory 生产，所有模块消费）。包含模型元数据、加速器选择和用于可变块大小的 `segmenter_kwargs`。
- `SpatialLayout`：不可变元数据（由 Segmenter 生产，由 Imputer 消费）
- `PhysicalMask`：具体张量掩码（由 Segmenter/imputer 转换生产，由 Masker 消费）
- `ProcessorOutput`：标准化模型输入（由 Factory 生产，由 Masker 和 Imputer 消费）

### 边界规则

| 规则 | 原因 |
|---|---|
| 只有 Imputer 可以触碰 `model` | 防止分散的 `model(**inputs)` 调用 |
| 只有 Imputer 拥有 `processor` 和原始 `input_image`/`input_text` | 预处理的唯一数据源 |
| `ImputerConfig` 在 Factory 创建后为只读 | 确保所有组件共享一致的模型元数据视图 |
| Segmenter 绝不在 `generate_masks` 内访问 GPU | "CPU 规划，GPU 执行"：掩码一次性生成，批量应用 |
| Masker 修改前克隆输入 | 防止跨批次迭代的变异 bug |
| Game 绝不导入 `torch` | 保持适配器纯净；所有张量操作在 ImputerFactory 中 |

---

## 数据传输协议

```
Coalitions (np.bool)                可视化 / Notebook
        │                                    │
        ▼                                    ▼
┌─────────────────┐              ┌─────────────────────┐
│   Segmenter     │              │  VisionLanguageGame  │
│  get_layout()   │──────────────│   （薄适配器）        │
│  generate_masks │              │  inputs / processor  │
└────────┬────────┘              │  value_function()    │
         │                       └──────────┬──────────┘
         ▼                                  │
   PhysicalMask                             │
         │                                  │
         ▼                                  ▼
┌─────────────────┐              ┌─────────────────────┐
│    Masker       │              │   ImageImputer      │
│  apply()        │◄─────────────│  forward_1d()       │
└────────┬────────┘              │  forward_crossmodal()│
         │                       └─────────────────────┘
         ▼
   ProcessorOutput（已修改） ───► model.forward() ───► np.array
```

---

## 端到端工作流（来自 `example.ipynb`）

以 CLIP ViT-B/32 解释 "black dog next to a yellow hydrant" 为例，完整追踪从原始输入到最终结果的数据流。

### 阶段 0：环境准备

```python
# ── 步骤 0.1：加载模型 ─────────────────────────────────────────────
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.to('cuda')
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ── 步骤 0.2：加载数据 ──────────────────────────────────────────────
input_text = "black dog next to a yellow hydrant"   # BOS/EOS 前 8 个 token
input_image = Image.open("assets/dog_and_hydrant.png")  # 224×224 RGB
```

此时还没有任何 Imputer 参与——只是加载了 HuggingFace 模型和 PIL 图像。

---

### 阶段 1：Factory 装配 Imputer

```python
factory = ImageImputerFactory()
imputer = factory.build(model, processor, input_image, input_text)
```

**内部过程：**

```
factory.build(model, processor, input_image, input_text)
│
├─ 1. _infer_model_type(model)
│     └─ model.name_or_path → "openai/clip-vit-base-patch32" → model_type = "clip"
│
├─ 2. 提取模型维度
│     └─ image_size=224, patch_size=32, n_channels=3
│
├─ 3. _preprocess(processor, image, text, "clip")
│     └─ processor(images=img, text=txt, return_tensors="pt", padding=True)
│     └─ 输出: {"pixel_values": (1,3,224,224), "input_ids": (1,10), "attention_mask": (1,10)}
│
├─ 4. _count_text_players(inputs_dict, "clip")
│     └─ CLIP: input_ids.size(1) - 2 = 10 - 2 = 8
│     └─ n_players_text = 8, text_total_length = 10
│
├─ 5. 构建 ImputerConfig
│     └─ ImputerConfig(model_type="clip", image_size=224, patch_size=32,
│         n_channels=3, n_players_image=49, n_players_text=8,
│         grid_size=7, text_total_length=10, accelerator=None,
│         segmenter_kwargs={})
│
├─ 6. _create_segmenter(config)
│     └─ PatchSegmenter(config)  ← 接收共享 config
│     └─ 生成 SpatialLayout(n_players_image=49, n_players_text=8, ...)
│
├─ 7. _create_masker(...)
│     └─ CrossModalMeanMasker()
│
├─ 8. 构建 ProcessorOutput + 装配 ImageImputer
│     └─ inputs_original: ProcessorOutput(pixel_values, input_ids, attention_mask)
│     └─ inputs_raw: 原始 HF dict（保留 .tokens() 等方法）
│     └─ config 传递给 ImageImputer  ← 所有组件共享
│     └─ input_image、input_text 保存在 imputer 上（用于 crossmodal 边缘情况）
│
└─ 返回: ImageImputer(model, processor, segmenter, masker, inputs_original, ...)
```

**装配结果：**
| 组件 | 实例 | 关键参数 |
|---|---|---|
| Segmenter | `PatchSegmenter` | grid=7×7, n_players_image=49 |
| Masker | `CrossModalMeanMasker` | 图像乘性掩码 + 文本注意力替换 |
| Layout | `SpatialLayout` | n_players=57（49 图像 + 8 文本） |
| Inputs | `ProcessorOutput` | 1 样本预处理结果 |

---

### 阶段 2：Game 封装

```python
game = VisionLanguageGame(imputer, batch_size=64)
```

**内部过程：**

```
VisionLanguageGame.__init__(imputer, batch_size=64)
│
├─ self.n_players_image = imputer.n_players_image  # 49
├─ self.n_players_text  = imputer.n_players_text   # 8
│
├─ 计算归一化基准值:
│   coalitions = [[False×57], [True×57]]  # 空联盟 + 全联盟
│   game_output = self.value_function(coalitions)
│     └─ 内部调用 imputer.forward_1d(coalitions, batch_size=64)
│     └─ 返回 [empty_value, full_value] = [-0.92, 27.3]
│
├─ self.empty_value = -0.92
├─ self.full_value  = 27.3
│
└─ super().__init__(n_players=57, normalize=True, normalization_value=-0.92)
    └─ shapiq.Game 会在每次 value_function 返回后自动减去 -0.92
```

**Game 提供的委托属性：**
- `game.inputs` → `imputer.inputs_raw`（HF BatchEncoding，支持 `.tokens()`）
- `game.processor` → `imputer.processor`（用于反标准化）

---

### 阶段 3：FIxLIP 初始化

```python
fixlip = FIxLIP(n_players_image=49, n_players_text=8, max_order=2,
                p=0.5, mode="banzhaf", random_state=0)
```

**内部过程：**

```
FIxLIP.__init__(n_players_image=49, n_players_text=8, max_order=2, p=0.5)
│
├─ mode="banzhaf" → 使用加权 Banzhaf 采样
├─ max_order=2 → 计算一阶归因 + 二阶交互
├─ is_crossmodal=True
│
├─ 创建 sampler_image: CoalitionSampler(n_players=49, ...)
│   └─ sampling_weights[k] = C(49,k) * p^k * (1-p)^(49-k)
│
├─ 创建 sampler_text: CoalitionSampler(n_players=8, ...)
│   └─ sampling_weights[k] = C(8,k) * p^k * (1-p)^(8-k)
│
└─ 总玩家数: n_players = 49 + 8 = 57
    交互种类数 = C(57,0) + C(57,1) + C(57,2) = 1 + 57 + 1596 = 1654
    其中 1596 种为二阶交互
```

---

### 阶段 4：计算解释

```python
interaction_values = fixlip.approximate_crossmodal(game, budget=2**19)
```

这是整个流程中最核心、最耗时的步骤。

**4.1 预算分配（`split_budget`）**

```
budget = 2^19 = 524,288
├─ n_players_text=8 < n_players_image=49
├─ budget_text = √524288 × 8/49 ≈ 115
├─ budget_image = 524288 / 115 ≈ 4559
└─ 最终: 4559 种图像 coalition × 115 种文本 coalition
```

**4.2 采样 Coalition**

```
sampler_image.sample(4559)  → coalitions_image: (4559, 49) bool
sampler_text.sample(115)    → coalitions_text:  (115, 8)   bool
```

每条 coalition 是一个 0/1 向量，表示哪些 patch/token 在场（1）或不在场（0）。

**4.3 评估 Coalition 值（`game.value_function_crossmodal`）**

```
coalitions_image (4559, 49)  ─┐
                               ├─► imputer.forward_crossmodal()
coalitions_text  (115, 8)    ─┘       │
                                       ▼
    ① Segmenter.generate_masks(coalitions_image, coalitions_text)
       └─ PhysicalMask:
          image_binary_mask: (4559, 3, 224, 224)  # patch 级 → 像素级展开
          text_attention_mask: (115, 10)           # BOS/EOS 自动补齐

    ② 双循环批处理 (batch_size=64):
       outer: 4559/64 ≈ 72 批图像
         inner: 115/64 ≈ 2 批文本
           ├─ Masker.apply() → pixel_values *= image_mask, attention_mask ← text_mask
           ├─ model(**masked_inputs) → logits_per_image
           └─ 收集 logits 矩阵

    ③ 返回: (4559, 115) np.array  ← 4559×115 个相似度分数
```

**批处理示意：**

```
              文本 coalition 1  文本 coalition 2  ...  文本 coalition 115
图像 coal 1    logit(1,1)        logit(1,2)             logit(1,115)
图像 coal 2    logit(2,1)        logit(2,2)             logit(2,115)
   ...
图像 coal 4559 logit(4559,1)     logit(4559,2)          logit(4559,115)
```

每格 = 一种（图像遮挡方式, 文本遮挡方式）组合下的模型相似度。

**4.4 归一化 + 重塑**

```python
coalition_values_crossmodal = (4559, 115) - game.normalization_value  # 减去 -0.92
# 重塑为 (4559×115,) = (524285,) 一维向量
coalition_values = coalition_values_crossmodal.reshape(-1)

# 把图像和文本 coalition 复制/平铺对齐
coalitions_matrix = (524285, 57)  # 49 图像位 + 8 文本位
├─ 前 49 列: sampler_image 的 coalition，每条重复 115 次
└─ 后 8 列:  sampler_text 的 coalition，整体平铺 4559 次
```

**4.5 加权回归 → 交互系数**

```
aggregate(coalition_matrix (524285, 57), coalition_values (524285,), regression_weights)
│
├─ interaction_lookup: {(): idx0, (0,): idx1, ..., (48,55): idx1653}
│   └─ 1654 个 interaction，其中 1596 个是二阶
│
├─ 构建回归矩阵 X (524285, 1654):
│   X[:, i] = coalition 中 interaction 的所有 player 同时为 1 时 = 1
│   例: interaction=(33,55) → coalition[33] × coalition[55]
│
├─ 加权最小二乘回归:
│   solve_regression(X, y, weights)
│   └─ 解得 φ[1654] ← 每个 interaction 的贡献系数
│
└─ 返回: InteractionValues(n_players=57, max_order=2, index="FWBII")
```

**回归原理：** coalition_value ≈ Σ φ_interaction × 𝟙[interaction ⊆ coalition]

即模型的相似度分数被分解为所有 interaction 的贡献之和。

---

### 阶段 5：结果输出

```python
print(interaction_values)
```

```
InteractionValues(
    index=FWBII, max_order=2, n_players=57,
    baseline_value=-0.9236,               # 空联盟值
    Top 10 interactions:
        (54,):     2.24   ← 一阶: token "black"（玩家 54）
        (33, 55):  1.57   ← 二阶: patch 33 × token "dog" 的协同
        (28, 50):  1.35   ← 二阶: patch 28 × token "yellow" 的协同
        (29, 50):  1.31   ← 二阶: 同上
        (36, 50):  1.26   ← 二阶: 同上
        (55, 56):  1.24   ← 二阶: token "dog" × "next" 的文本内交互
        (52,):    -1.39   ← 一阶: token 52 有负贡献
        (56,):    -1.76   ← 一阶
        (50,):    -4.61   ← 一阶: token "yellow" 的强负贡献
        (55,):    -5.68   ← 一阶: token "dog" 的强负贡献
)
```

**解读：**
- **(54,): +2.24** — "black" 这个 token 单独贡献 +2.24 的相似度
- **(33, 55): +1.57** — patch 33 和 "dog" 同时存在时，额外产生 +1.57 的协同效应
- **(50,): -4.61** — "yellow" 单独在场时反而降低相似度（可能是因为图像中有黄色的消防栓，但 "yellow" 这个单词需要和其他 token 配合才能正确定位）

---

### 阶段 6：可视化

```python
src.plot.plot_image_and_text_together(
    img=input_image_denormalized,
    text=text_tokens,           # ['black', 'dog', 'next', 'to', 'a', 'yellow', 'hydrant', '']
    image_players=list(range(49)),
    iv=interaction_values,
    plot_interactions=True,     # 绘制二阶交互连线
    top_k=14,                   # 展示 top-14 交互
    normalize_jointly=True,     # 联合归一化一阶+二阶
)
```

**可视化元素：**
- **图像侧热力图**：一阶归因（每个 patch 的单独贡献）
- **文本侧条形图**：一阶归因（每个 token 的单独贡献）
- **跨模态连线**：二阶交互（patch ↔ token 之间的彩色线，宽度/颜色 = 交互强度）
- **文本内连线**：二阶交互（token ↔ token 之间的交互）

---

### 完整流程图

```
  输入层               装配层                      计算层                    输出层
┌────────┐    ┌──────────────────┐    ┌─────────────────────┐    ┌──────────────┐
│ Image  │    │ ImageImputer     │    │ FIxLIP              │    │ (54,): 2.24  │
│ Text   │───►│ Factory.build()  │    │ .approximate_       │    │ (33,55):1.57 │
│ Model  │    │                  │    │  crossmodal(game,   │    │ (28,50):1.35 │
│Proc.   │    │ ┌PatchSegmenter┐ │    │  budget=2^19)       │    │    ...       │
└────────┘    │ │CrossModal    │ │    │                     │    └──────────────┘
              │ │MeanMasker    │ │    │ ① split_budget     │
              │ └──────────────┘ │    │ ② sample coalitions │
              │         │        │    │ ③ game.value_func_  │
              │         ▼        │    │   crossmodal()      │
              │ ┌ImageImputer┐  │    │   └→ imputer.fwd_   │    ┌──────────────┐
              │ │forward_1d() │  │    │      crossmodal()   │    │  热力图 +    │
              │ │fwd_cross()  │◄─┼────┤ ④ aggregate(WLS)   │───►│  交互连线    │
              │ └────────────┘  │    └─────────────────────┘    │              │
              └──────────────────┘                              └──────────────┘
```

---

## 关键设计决策

1. **"CPU 规划，GPU 执行"**：Segmenter 在 CPU 上一次性生成整数索引映射。成千上万次 coalition→mask 转换完全在 GPU 上通过原生张量操作完成。

2. **ImputerConfig 作为共享事实**：`ImputerConfig` 由 Factory 一次性创建，在 Segmenter、Masker 和 Imputer 之间只读共享。消除了分散的模型内省。`segmenter_kwargs` 携带可变块大小参数从 Factory → Segmenter。

3. **Imputer 拥有输入**：`ImageImputer` 存储 `inputs_original`（ProcessorOutput）、`inputs_raw`（HF dict，用于 `.tokens()`）以及 `input_image`/`input_text`（用于 crossmodal 批量大小不一致的边缘情况）。

4. **Game 是薄外壳**：`VisionLanguageGame` 将所有掩码/批处理/模型前向传播委托给 Imputer。它只处理 shapiq 调度（归一化值、玩家计数）。

---

## 当前实现细节

### `ImputerFactory/data.py`
四个 dataclass 作为通用数据协议：
- **`ImputerConfig`**：由 Factory 生产的只读配置。包含 `model_type`、`image_size`、`patch_size`、`n_channels`、`n_players_image`、`n_players_text`、`grid_size`、`text_total_length`、`accelerator` 和 `segmenter_kwargs`（传递给 Segmenter 用于可变块大小）。所有组件共享。
- **`SpatialLayout`**：描述空间划分的不可变元数据。由 Segmenter 一次性生产，由 Imputer 消费。
- **`PhysicalMask`**：具体张量掩码。`image_binary_mask` (N, C, H, W) float + `text_attention_mask` (N, L) int。
- **`ProcessorOutput`**：封装 `pixel_values`、`input_ids`、`attention_mask`，提供 `to_dict()` 用于模型前向传播。

### `ImputerFactory/segmenters/patch.py` — PatchSegmenter
- 构造函数：`PatchSegmenter(config: ImputerConfig)` — 接收共享配置
- 所有模型元数据（image_size、patch_size 等）从 `config` 读取，不直接访问模型
- 初始化时预计算 `SpatialLayout`（is_stateful=False）
- `generate_masks()` 将 coalition 数组转换为 `PhysicalMask`
- 图像：将 patch 级布尔值展开为 `patch_size×patch_size` 块 → (N, C, H, W)
- 文本：处理 CLIP（BOS/EOS 包裹）与 SigLIP（右填充）两种掩码格式

### `ImputerFactory/maskers/mean.py` — CrossModalMeanMasker
- 图像：`pixel_values *= image_binary_mask`（零均值归一化 → 均值填充）
- 文本：用 coalition 生成的掩码替换 `attention_mask`
- 克隆输入以避免变异

### `ImputerFactory/core/imputer.py` — ImageImputer
- 构造函数：接收 `config: ImputerConfig` — 模型元数据从 config 派生，而非直接从 `model` 获取
- **`forward_1d(coalitions, batch_size)`**：拆分 coalition → 生成掩码 → 分批 → 掩码 → 模型 → 提取对角线
- **`forward_crossmodal(coalitions_img, coalitions_txt, batch_size)`**：双循环（图像外层，文本内层）。边缘情况：当 txt_bs ≠ img_bs 时，使用存储的 `input_image`/`input_text` 通过 `_preprocess_batch()` 重新处理
- **`_model_forward()`**：自动检测模型设备，前向传播前移动输入
- 存储：`config`、`inputs_original`、`inputs_raw`、`input_image`、`input_text`、`model`、`processor`、`segmenter`、`masker`、`layout`

### `ImputerFactory/factory.py` — ImageImputerFactory
- `build(model, processor, input_image, input_text, accelerator=None)`：
  1. 推断模型类型（clip/siglip/siglip2）
  2. 提取模型维度并计算派生值（grid_size、n_players_image）
  3. 一次性预处理以确定 `n_players_text` + `text_total_length`
  4. **构建 `ImputerConfig`** — 所有元数据、加速器选择和 `segmenter_kwargs` 的唯一数据源
  5. 通过 `_create_segmenter(config)` 创建 segmenter — config 流入 Segmenter
  6. 创建 `CrossModalMeanMasker`
  7. 将 `ProcessorOutput` + 原始 dict + config + 原始图像/文本注入 `ImageImputer`

### `Game/game_huggingface.py` — VisionLanguageGame
- 构造函数：`VisionLanguageGame(imputer, batch_size=64, verbose=False)`
- `n_players_image` / `n_players_text` 来自 imputer layout
- `inputs` / `processor` 属性委托给 imputer（向后兼容）
- `value_function()` → `imputer.forward_1d()`
- `value_function_crossmodal()` → `imputer.forward_crossmodal()`

---

## 团队分工与任务规划

> 团队规模：4 名工程师 + 1 名 PM。分为两个专职小组。

### 团队结构

```
┌─────────────────────────────────┐
│         PM（1 人）               │
│  跨组协调                        │
│  需求与优先级管理                 │
└──────────┬──────────────────────┘
           │
   ┌───────┴───────┐
   ▼               ▼
┌──────────┐  ┌──────────────┐
│  A 组    │  │   B 组       │
│  测试与  │  │   功能开发   │
│  模型适配 │  │   与 Bug 修复 │
│ （2 人）  │  │  （2 人）    │
└──────────┘  └──────────────┘
```

---

### A 组 — 测试与模型适配（2 人）

**使命**：确保所有实验通过，Imputer + Game 管道在 CLIP 模型变体上正确工作。向 B 组报告阻塞问题。

#### A1. 实验迁移与验证

| # | 目标 | 详情 | 成功标准 |
|---|---|---|---|
| A1.1 | `experiments/faithfulness.py` | 迁移到 `Game.game_huggingface` API | 与 `src` 基线保持相同的忠实度指标（±1e-4） |
| A1.2 | `experiments/insertion_deletion.py` | 迁移到 `Game.game_huggingface` API | 与 `src` 基线保持相同的 AID 曲线 |
| A1.3 | `experiments/insertion_deletion_siglip.py` | 迁移并验证 SigLIP 支持 | 正确检测模型类型，无崩溃 |
| A1.4 | `experiments/pointing_game_banzhaf.py` | 迁移到 `Game` API | 保持相同的 PGR 准确率 |
| A1.5 | `experiments/pointing_game_shapley.py` | 迁移到 `Game` API | 保持相同的 PGR 准确率 |
| A1.6 | `experiments/pointing_game_crossmodal.py` | 迁移到 `Game` API | 保持相同的 PGR 准确率 |
| A1.7 | `experiments/explain_mscoco.py` | 迁移到 `Game` API | 保持相同的 top-k 交互重叠率 |
| A1.8 | `experiments/explain_mscoco_siglip.py` | 迁移并验证 SigLIP2 支持 | SigLIP2 模型加载并运行 |

#### A2. 数值等效性回归测试

| # | 任务 | 详情 |
|---|---|---|
| A2.1 | 构建比较工具 | 脚本通过 `src` Game 和 `Game` Game 运行相同 coalition，对比输出差异 |
| A2.2 | 快照基线 | 使用 `src` 路径保存所有 8 个实验的参考输出 |
| A2.3 | CI 风格门禁 | 如果任何实验与基线偏差 > 1e-4，退出码 ≠ 0 |

#### A3. 跨模型适配测试

| # | 任务 | 详情 |
|---|---|---|
| A3.1 | CLIP ViT-B/32 | 已在 `example.ipynb` 中验证 |
| A3.2 | CLIP ViT-B/16 | 测试 196 个图像玩家（14×14 网格） |
| A3.3 | CLIP ViT-L/14 | 测试 256 个图像玩家（16×16 网格），验证内存使用 |
| A3.4 | SigLIP base-patch16 | 测试 model_type 检测 + 文本掩码逻辑 |
| A3.5 | SigLIP2 so400m | 测试 model_type 检测（`siglip2` 路径） |

#### A4. 向 B 组的反馈回路

- 提交带有最小复现脚本的 bug 报告
- 标记 API 粗糙点（例如 `inputs` / `processor` 委托模式）
- 报告与 `src` 基线相比的性能回归

---

### B 组 — 功能开发与 Bug 修复（2 人）

**使命**：实现 CLIP 兼容的加速器 Segmenter，修复 A 组报告的 bug，优化 Imputer 管道。

#### B1. Bug 修复（响应式——来自 A 组报告）

| # | 类别 | 预期来源 |
|---|---|---|
| B1.1 | 设备放置 | 边缘情况下的 CPU/CUDA 不匹配 |
| B1.2 | Crossmodal 批量大小 | txt_bs ≠ img_bs 的正确性 |
| B1.3 | 模型类型检测 | 边界模型名称模式 |
| B1.4 | 内存 / OOM | 大模型（ViT-L）高预算 |

#### B2. 加速器 Segmenter

| # | 功能 | 详情 | 优先级 |
|---|---|---|---|
| B2.1 | `GradientGuidedSegmenter` | 提取梯度图 → skimage 分水岭 → 非均匀静态布局 | 中 |
| B2.2 | `AdaptiveSegmenter` | 粗网格 → 分数驱动细分 → 反馈循环。需要 Imputer ↔ Segmenter 之间的 `is_stateful=True` 协议 | 中 |

#### B3. Masker 扩展

| # | 功能 | 详情 |
|---|---|---|
| B3.1 | `AttentionMasker` 实现 | Hook 自注意力，注入 -inf 掩码矩阵。需要 PyTorch `register_forward_hook` 或 HF `output_attentions` 覆写 |

#### B4. 后端适配器提取

| # | 功能 | 详情 |
|---|---|---|
| B4.1 | `TorchOps` 提取 | 将 Imputer/Segmenter 中的内联 PyTorch 操作移入适配器 |
| B4.2 | `JaxOps` 骨架 | 为 JAX 原生模型提供接口 + 占位 |

#### B5. 性能优化

| # | 任务 | 详情 |
|---|---|---|
| B5.1 | `_repeat_inputs` 内存 | 用 stride tricks 替换 `.expand().clone()` |
| B5.2 | AMP 支持 | `torch.autocast` 用于混合精度前向传播 |

---

### PM — 协调与监督（1 人）

| # | 职责 |
|---|---|
| P1 | 维护本文档作为唯一事实来源 |
| P2 | 周度同步：A 组报告阻塞项 → B 组优先修复 |
| P3 | 分诊 A4 bug 报告，分配严重级别，跟踪解决 |
| P4 | 审查 API 决策（命名、数据格式、公共接口） |
| P5 | 签收实验迁移检查点（A1.1–A1.8） |
| P6 | 维护比较工具（A2.1）作为合并的门禁 |

---

### 任务依赖

```
A 组                                 B 组
──────                              ──────
A1.1–A1.8（迁移实验）              B2.1 GradientGuidedSegmenter
    │                                   │
    ├─ A2（等效性测试）──────────────────┤（bug 报告）
    │       │                           │
    │       ▼                           ▼
    ├─ A3（跨模型）──────────────► B1（bug 修复）
    │       │                           │
    │       ▼                           ▼
    └─ A4（反馈）───────────────► B2.2 AdaptiveSegmenter
                                        │
                                        ▼
                                    B3–B5（扩展）
```

| 依赖 | 阻塞项 | 被阻塞于 |
|---|---|---|
| B1（bug 修复） | A2/A3 报告 | A 组发现 |
| B2.2（Adaptive） | `is_stateful` 协议 | B1 稳定性 |
| A3（跨模型） | A1 完成 | 所有实验通过 |

---

### 里程碑时间表

| 周 | A 组 | B 组 | PM 门禁 |
|---|---|---|---|
| W1 | A1.1–A1.4, A2.1 | B1（bug 修复）, B4.1（TorchOps） | 实验 1–4 通过 |
| W2 | A1.5–A1.8, A2.2–A2.3 | B2.1（GradientGuided） | 8 个实验全部通过 |
| W3 | A3.1–A3.5（跨模型） | B2.2（Adaptive）, B3.1 | 跨模型测试绿色 |
| W4 | A4（反馈回路） | B4.2（JaxOps）, B5.1（内存） | 功能冻结，集成测试 |

---

### 已知问题（B1 跟踪）

- **Crossmodal 边缘情况的 processor 调用**：当 `budget_image % batch_size ≠ 0` 或 `budget_text % batch_size ≠ 0` 时，最后一批图像和/或文本的大小不足（例如 `batch_size=64, budget_image=4559, budget_text=115` 时 `img_bs=15, txt_bs=51`）。2 种图像批 × 2 种文本批 = 4 种组合中有 3 种情况下 `img_bs ≠ txt_bs`，此时 `_preprocess_batch()` 必须重新调用 HF processor 创建 batch 维度匹配的输入。原始 `src` 代码有相同行为（在等价分支中直接调用 `processor_function`），因此这不是回归——而是双循环 crossmodal 设计的固有特性。每次 `forward_crossmodal` 的额外调用最多 3 次（每次约 2 ms，可忽略）。
- `_repeat_inputs` 使用 `.expand().clone()` 复制内存；可用 stride tricks 优化
- 尚不支持混合精度（AMP）——对更大模型有影响
