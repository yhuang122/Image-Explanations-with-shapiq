## ImageImputer Architecture Overview

### ImageImputerFactory
The **ImageImputerFactory** serves as the central assembly line, finalizing the blueprint, instantiating required components, and injecting them into the `ImageImputer`. It establishes necessary feedback loops (required by adaptive accelerators) and returns a fully optimized, ready-to-run execution container.

---

### 1. Backend Adapters
These adapters abstract all tensor manipulations into a `TensorOps` interface.
*   **Concrete Implementations:** `TorchOps` and `JaxOps` encapsulate framework-specific behaviors.
*   **Purpose:** Ensures that upper-level business logic remains completely framework-agnostic.

### 2. Segmenters
Responsible for spatial division of the input data.
*   **PatchSegmenter:** Uses rigid grids, mathematically aligned with Vision Transformers (ViTs).
*   **SLICSegmenter:** Uses perceptual superpixels, preserving natural edges to prevent Out-of-Distribution (OOD) artifacts for CNNs.
*   **GradientGuidedSegmenter:** 
    *   Acts as a gradient-to-layout translator. 
    *   Ingests a pre-computed backpropagation gradient map to generate a static, non-uniform segmentation layout. 
    *   High-gradient regions are finely partitioned, while near-zero zones are consolidated into massive background blocks. 
    *   Relies solely on heuristics and does not participate in the evaluation loop.
*   **AdaptiveSegmenter:** 
    *   Implements a coarse-to-fine, score-driven spatial division. 
    *   Initializes with a uniform coarse grid and dynamically subdivides high-scoring regions based on intermediate Shapley attribution scores. 
    *   Freezes low-scoring blocks to aggressively prune the combinatorial search space.
*   **HybridSegmenter:** 
    *   The structural synthesis utilizing the **Composition pattern**. 
    *   Uses the `GradientGuidedSegmenter` to generate an intelligent initial layout rather than a blind grid. 
    *   Feeds this baseline into the `AdaptiveSegmenter` logic to refine critical regions as empirical Shapley scores are computed. 
    *   Marries rapid gradient localization with rigorous perturbation fairness.

### 3. Maskers
Responsible for feature occlusion.
*   **MeanMasker:** Injects average pixel values directly into the input tensor.
*   **AttentionMasker:** Intercepts the model's internal self-attention mechanism by generating negative-infinity mask matrices (via JAX/HuggingFace parameters or raw PyTorch internal hooks).

---

### 4. Core Orchestration (`ImageImputer`)
*   **Layout Request & Feedback Loop:** Requests the spatial layout from the Segmenter. For stateful segmenters (Adaptive/Hybrid), the Orchestrator feeds previous Shapley evaluation results back to the Segmenter to refine the next layout.
*   **Translation:** Translates the binary Shapley coalition array into a physical tensor mask based on the active layout.
*   **Execution:** Directs the Masker to apply the physical mask and executes the model forward pass with modified inputs/kwargs.

---

### 5. The Assembly Line (`ImageImputerFactory`)
The Factory manages the construction of the imputer through the following steps:

1.  **Backend Selection:** The user provides a pre-trained model and specifies the backend (`"pytorch"` or `"jax"`).
2.  **Routing & Accelerator Selection:** Inspects the model to establish a functional baseline and injects high-performance spatial engines based on configuration.
3.  **Baseline Configuration:** Infers default components:
    *   **Transformers:** `PatchSegmenter` + `AttentionMasker`.
    *   **CNNs:** `SLICSegmenter` + `MeanMasker`.
4.  **Accelerator Selector:** An optional `accelerator` parameter allows for overriding the baseline to speed up Shapley convergence:
    *   `gradient`: Injects `GradientGuidedSegmenter` for a static, non-uniform spatial prior.
    *   `adaptive`: Injects `AdaptiveSegmenter` for dynamic, coarse-to-fine sub-grid exploration.
    *   `hybrid`: Injects `HybridSegmenter` for maximum pruning efficiency via gradient priors and stateful evaluation.