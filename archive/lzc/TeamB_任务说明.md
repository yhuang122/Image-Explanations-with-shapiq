# Team B 任务说明

> 来源：ImplementationProgressSummary.md，2026-05-13 更新

---

## 你在项目里的位置

```
        PM（1人，组长）
       /              \
    Team A（2人）      Team B（2人）
    QA & 模型适配      功能开发 & 修Bug
```

Team B 的定位：**实现新功能、修 Team A 报的 Bug、优化性能**。

---

## 项目当前状态

组员已经写好了核心架构（✅ Done）：
- **数据管道**：ImputerConfig、SpatialLayout、PhysicalMask、ProcessorOutput
- **PatchSegmenter**：均匀网格切图（7×7=49 个 patch）
- **CrossModalMeanMasker**：用均值灰度遮住不用的 patch
- **ImageImputer**：串联整个流程（前向推理 + 批量处理）
- **Factory**：自动检测模型类型，组装管道
- **VisionLanguageGame**：把管道包装成 shapiq 的 Game 接口

你们要填的 3 个空壳模块（⏳ Stub）：

---

## 任务 1：GradientGuidedSegmenter（B2.1，W2）

```
当前代码（adaptive.py，全是空壳）：
class GradientGuidedSegmenter(BaseSegmenter):
    """Static non-uniform layout using gradients."""
    def __init__(self, config):
        super().__init__(config)   # 什么都没有
```

### 要做什么

- 对输入图片做一次**前向 + 反向传播**，提取梯度图
- 梯度大 = 模型重点关注的区域 → 切成**小而密集的 patch**
- 梯度小 = 模型不关心的背景 → 用**大块粗切**
- 用 `skimage.segmentation.watershed` 做非均匀分割
- 效果：狗脸区域 patch 细碎，天空/背景 patch 粗大

### 你需要掌握的库
- `torch.autograd`（梯度提取）
- `skimage.segmentation.watershed`（分水岭算法）
- `skimage.filters.sobel` 之类的边缘检测

---

## 任务 2：AdaptiveSegmenter（B2.2，W3）

```
当前代码（adaptive.py，也是空壳）：
class AdaptiveSegmenter(BaseSegmenter):
    """Dynamic, coarse-to-fine scoring-driven spatial division."""
    def __init__(self, config):
        super().__init__(config)   # 什么都没有
```

### 要做什么

- 先粗切（比如 4×4 = 16 个大块）
- 对每个大块算 FIxLIP 分数（它对图文匹配贡献多大）
- 贡献大的块（狗脸）→ **继续切细**（一分为四）
- 贡献小的块（背景）→ **保持不动**
- 迭代几轮，最终得到"重点密、非重点疏"的自适应布局
- 关键：需要 `is_stateful=True` 协议 —— Segmenter 要记住上一轮的分数，不能是无状态的

### 你需要理解的概念
- 递归细分（recursive subdivision）
- shapiq 的 InteractionValues（怎么读每个 patch 的贡献值）
- `is_stateful` 协议：Imputer 和 Segmenter 之间的状态传递

---

## 任务 3：AttentionMasker（B3.1，W3）

```
当前代码（attention.py，纯空壳）：
class AttentionMasker(BaseMasker):
    """Intercepts self-attention handling -infinity masks."""
    pass
```

### 要做什么

当前 CrossModalMeanMasker 的做法：不想让模型看某个 patch → 用均值灰度**替换掉**
AttentionMasker 的做法：不想让模型看某个 patch → 在 ViT 的 self-attention 矩阵里注入 **-inf**，让 attention 权重 = 0

这更高档，因为：
- 不会产生 OOD（Out-of-Distribution）的人工痕迹（灰色块）
- 是 ViT 原生的屏蔽机制

### 你需要掌握的库
- PyTorch `register_forward_hook`（钩住 attention 层）
- ViT 的 attention 机制（Q、K、V、attention matrix）

---

## 杂项任务

| 编号 | 干什么 | 难度 | 时间 |
|------|--------|------|------|
| B4.1 TorchOps 提取 | 把散落各处的 `.to(device)` 收进一个适配器类 | 低（重构） | W1 |
| B5.1 内存优化 | `.expand().clone()` → stride tricks，省显存 | 低 | W4 |
| B5.2 AMP 混合精度 | 加 `torch.autocast` 加速大模型推理 | 低 | W4 |
| B4.2 JaxOps 空壳 | 为 JAX 模型留接口 | 低（写空壳） | W4 |

---

## 时间线

| 周 | 你要做的 | 状态 |
|----|----------|------|
| W1 | B1 修 Bug（等 A 组报告）+ B4.1 TorchOps | 等 A 组开工 |
| W2 | B2.1 GradientGuidedSegmenter | 未开始 |
| W3 | B2.2 AdaptiveSegmenter + B3.1 AttentionMasker | 未开始 |
| W4 | B4.2 JaxOps + B5.1 内存优化 | 未开始 |

---

## 和队友分工建议

- **一人做 GradientGuided**（偏图像处理，梯度/watershed）
- **一人做 Adaptive**（偏算法逻辑，递归细分/状态管理）
- AttentionMasker 和优化任务谁有空谁搞

你自己选想做的方向，然后跟队友说。
