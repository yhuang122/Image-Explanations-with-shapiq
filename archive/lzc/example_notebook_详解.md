# example.ipynb 逐步详解

## Cell 1: Import（导入库）

```python
import torch
# torch.set_float32_matmul_precision("high")  # 被注释掉了
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import matplotlib.pyplot as plt
import shapiq 
import src
```

- `torch`：PyTorch 深度学习框架
- `transformers`：HuggingFace 的模型库，提供 CLIP 模型和处理器
- `PIL.Image`：读取图片
- `matplotlib`：画图
- `shapiq`：Shapley Interaction Quantification，计算 Shapley/Banzhaf 交互值的库
- `src`：FIxLIP 项目自己的代码（近似器、画图工具、评估函数等）
- 被注释的那行 `torch.set_float32_matmul_precision("high")` 是让 GPU 用 TF32 加速矩阵乘法。在 Windows Jupyter kernel 里容易导致崩溃，注释掉不影响计算结果，只慢一点点

---

## Cell 2: 加载 CLIP 模型

```python
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.to('cuda')
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
```

- **CLIPModel**：OpenAI 训练的图文匹配模型
  - ViT-B/32：Vision Transformer，Base 规模，patch size = 32×32 像素
  - 能把图片映射到一个向量，文字映射到同一个空间，然后算**余弦相似度**
- **model.to('cuda')**：把模型参数搬到 GPU 上，推理加速
- **CLIPProcessor**：CLIP 的"数据预处理器"，负责：
  - 图片 → 调整大小、归一化 → `pixel_values` 张量
  - 文字 → tokenize → `input_ids` 张量

---

## Cell 3: 加载数据

```python
input_text = "black dog next to a yellow hydrant"
input_image = Image.open("assets/dog_and_hydrant.png")
```

- 图片里：一只黑狗 + 一个黄色消防栓
- 文字是对图片内容的描述
- 目标：解释 CLIP 为什么认为这句话和这张图是匹配的

---

## Cell 4: 定义 Game（合作游戏）—— 最核心的概念

```python
from ImputerFactory import ImageImputerFactory
from Game.game_huggingface import VisionLanguageGame

factory = ImageImputerFactory()
imputer = factory.build(model, processor, input_image, input_text)
game = VisionLanguageGame(imputer, batch_size=64)
n_players_image = game.n_players_image  # 49
n_players_text = game.n_players_text    # 8
```

### 什么是"Game"（合作游戏）？

Shapley/Banzhaf 解释的核心思想：**把模型预测看成一个合作游戏**

- **玩家（Player）**：输入的每个组成部分
  - 图片被 ViT 切成 **7×7 = 49 个 patch**（每个 32×32 像素的格子）
  - 文字 "black dog next to a yellow hydrant" 被 tokenizer 切成 **8 个 token**
  - 总共 **57 个玩家**
- **联盟（Coalition）**：任意玩家子集。比如 "只保留左上角 10 个 patch + 前 2 个 token"
- **价值函数 value_function(coalition)**：给定一个 coalition，让模型只"看到"这些玩家，输出图文相似度分数

### 管道架构（ImputerFactory → ImageImputer）

```
原始图片 + 文字
      ↓
CLIPProcessor（预处理）
      ↓
PatchSegmenter（负责把 coalition → 像素掩码）
      ↓
CrossModalMeanMasker（负责"遮住"不在 coalition 里的部分）
      ↓
CLIP Model（前向推理）
      ↓
余弦相似度分数
```

- **Segmenter**：把 coalition 数组（49 个 0/1，表示哪些 patch 参与）转成像素级二值掩码
- **Masker**：图像用二值掩码遮住不参与的区域（替换为均值灰度），文字用 attention_mask 屏蔽不参与的 token
- **Imputer**：协调整个流程，做批量推理

`VisionLanguageGame` 把这个流程包装成 `value_function()` 接口，供 shapiq 的 Approximator 调用。

---

## Cell 5: 定义 FIxLIP 近似器

```python
fixlip = src.fixlip.FIxLIP(
    n_players_image=49,
    n_players_text=8, 
    max_order=2,
    p=0.5,
    mode="banzhaf",
    random_state=0
)
```

- **max_order=2**：算一阶（单个玩家贡献）+ 二阶（两两交互）
  - 一阶：比如 "black 这个词单独贡献了多少"
  - 二阶：比如 "黑狗图片区域 × black 这个词，两者在一起是互相增强还是减弱？"
- **p=0.5**：Weighted Banzhaf 的权重参数。p=0.5 是平衡点，给高阶交互适中的关注
- **mode="banzhaf"**：用 Banzhaf 值而非 Shapley 值
  - Banzhaf 计算更快（不需要 Shapley 的阶乘权重）
  - 理论上等价但采样策略不同
- FIxLIP 的论文卖点：不用暴力枚举全部 2^57 ≈ 10^17 种组合，而是用**稀疏傅里叶变换 + 巧妙采样**，用约 52 万次查询逼近真实值

---

## Cell 6: 计算解释（最耗时，约 1 分钟）

```python
src.utils.set_seed(0)
interaction_values = fixlip.approximate_crossmodal(game, budget=2**19)
```

- `budget=2**19 = 524,288`：总共进行约 52 万次 coalition 查询
- **跨模态采样**：图片 coalition 和文字 coalition 分开采样，用双循环（外层图片、内层文字）
- 每次查询的流程：
  1. 取一个 coalition 向量（57 维 0/1）
  2. Segmenter 把图片部分转成像素掩码
  3. Masker 遮住不在 coalition 里的 patch（用均值填充）和 token（attention mask 屏蔽）
  4. CLIP 前向推理
  5. 返回图文相似度分数

---

## Cell 8: 查看结果

```
InteractionValues(
    index=FWBII, max_order=2, min_order=0, estimated=True,
    n_players=57, baseline_value=-0.9236,
    Top 10 interactions:
        (54,): 2.2405       ← 玩家 54 单独贡献最大
        (33, 55): 1.5663    ← 玩家 33（某图片patch）× 玩家 55（某token）的正向交互
        (55, 56): 1.2403    ← 相邻 token 的正向交互
        (50,): -4.6092      ← 玩家 50 的单独贡献是负的
        (55,): -5.6804      ← 玩家 55 的单独贡献是负的
)
```

解读：
- **baseline_value = -0.9236**：空 coalition（所有玩家都不参与）时的分数
- **正值**：该玩家/交互让图文相似度**变高**（正向贡献）
- **负值**：该玩家/交互让图文相似度**变低**（负向贡献）
- `(33, 55): 1.5663` 这种带括号的就是**二阶交互**，表示玩家 33 和 55 在一起时产生了额外的增强效果

**关键洞察**：有些玩家单独看贡献是负的（如玩家 55 的 -5.68），但和其他玩家交互时是正的（如 33×55 的 +1.57）。这说明只看单个玩家的贡献（一阶）会漏掉重要的交互信息，FIxLIP 的二阶解释能捕捉这些。

---

## Cell 9: 辅助函数

```python
image_mean = (0.48145466, 0.4578275, 0.40821073)
image_std = (0.26862954, 0.26130258, 0.27577711)
```

CLIP 输入的图片是归一化过的（减均值除方差），画图时要反归一化回去才能正常显示。

---

## Cell 10: 准备可视化数据

```python
text_tokens = game.inputs.tokens()
text_tokens = text_tokens[1:-1]  # 去掉开头 BOS 和结尾 EOS
text_tokens = [token.replace('</w>', '') for token in text_tokens]
input_image_denormalized = ...  # 反归一化图片
```

- 从 CLIP 的 tokenizer 输出中提取原始文本 token
- 去掉 BOS（Begin of Sequence）和 EOS（End of Sequence）这两个特殊标记
- `</w>` 是 CLIP tokenizer 的子词标记（表示这是一个完整词的结尾），去掉后显示更干净
- 反归一化图片，让像素值回到正常范围 [0, 1]

---

## Cell 11: FIxLIP 可视化（核心输出）

```python
src.plot.plot_image_and_text_together(
    img=input_image_denormalized,
    text=text_tokens,
    image_players=list(range(49)),
    iv=interaction_values,
    plot_interactions=True,
    top_k=14,
    ...
)
```

画出图文联合解释热力图：

**图像部分（左半）**：
- 每个 32×32 patch 覆盖一个半透明色块
- 红色越深 = 该区域对匹配贡献越大
- 蓝色越深 = 该区域贡献为负

**文字部分（右半）**：
- 每个 token 有对应的色条
- 同样红色=正贡献，蓝色=负贡献

**交互连线**：
- 最核心的 feature：展示图片 patch ↔ 文字 token 之间的交互强度
- 比如 "黑狗所在格子" ↔ "black" 之间应该有一条粗线
- `(33, 55): 1.5663` 就会显示为 patch 33 和 token 55 之间的连线

这就是 FIxLIP 论文里最核心、最有辨识度的图。

---

## Cell 12-27: 对比实验

加载了三种对比方法的结果：

| 方法 | max_order | 原理 |
|------|-----------|------|
| **GAME** | 1 | 只算单玩家贡献，无交互信息 |
| **Grad-ECLIP** | 1 | 用梯度近似每个玩家的重要性 |
| **exCLIP** | 2 | 另一种计算交互的方法（和 FIxLIP 算法不同） |

每种方法都画了热力图，用于对比。FIxLIP 的优势：
1. 相比 GAME/Grad-ECLIP：能捕捉**二阶图文交互**
2. 相比 exCLIP：采样效率更高，结果更稳定

---

## 关键概念速查

| 概念 | 含义 |
|------|------|
| Player（玩家） | 输入的每个最小单元（一个 patch 或一个 token） |
| Coalition（联盟） | 玩家的任意子集 |
| Value Function（价值函数） | 给定 coalition，输出模型分数 |
| Shapley Value | 每个玩家的"公平贡献" |
| Banzhaf Value | 另一种分配方案，计算比 Shapley 快 |
| k-SII | k 阶 Shapley 交互：1 阶=个人贡献，2 阶=两两交互 |
| FIxLIP | Faithful Interaction Explanations of CLIP — 用 Weighted Banzhaf 做图文解释 |
