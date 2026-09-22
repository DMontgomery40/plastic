# Test-Time Training Layers, Fast Weights, and SSM Fusion: State of the Art as of 2026-09-21

Every entry was checked against its arXiv abstract page (dates are arXiv v1 submission dates unless noted) or the cited primary page; equations are transcribed from the papers' HTML versions. URL convention: arXiv:NNNN.NNNNN is at https://arxiv.org/abs/NNNN.NNNNN. Non-arXiv sources are linked inline.

## 0. The unifying picture

Since early 2025 the field has converged on one reading of linear-time sequence layers: the recurrent state is a set of fast weights M trained online, at test time, to fit an in-context regression from keys to values, and the output is the query pushed through those weights. Wang, Shi and Fox (test-time regression) and Behrouz et al. (MIRAS) formalize this. Linear attention is one gradient step on an unnormalized least-squares loss; DeltaNet and Gated DeltaNet are online SGD on ½||M k − v||²; Mamba-2 and GLA are exponentially weighted variants; MesaNet solves the least squares exactly; softmax attention is the nonparametric (Nadaraya-Watson) solution; TTT-Linear, TTT-MLP, Titans and LaCT use nonlinear (MLP) fast weights and richer optimizers. The 2026 literature splits three ways: better inner-loop optimizers for linear fast weights; end-to-end TTT where the inner loss is the model's own next-token loss and the system is meta-trained; and engineering showing that large-chunk updates with Muon and normalized weights make nonlinear fast weights train stably.

## 1. TTT layers (Sun et al., 2024)

**Learning to (Learn at Test Time): RNNs with Expressive Hidden States.** Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, Carlos Guestrin. arXiv:2407.04620, submitted 2024-07-05 (v2 2025-08-31).

Mechanism. The hidden state is the weight matrix W of a small model f, and the recurrence is one step of self-supervised gradient descent per token. With learnable low-rank projections θ_K, θ_V, θ_Q (training view, label view, test view):

```
ℓ(W; x_t) = ||f(θ_K x_t; W) − θ_V x_t||²
W_t = W_{t−1} − η(x_t) ∇ℓ(W_{t−1}; x_t),    η(x) = η_base · σ(θ_lr · x)
z_t = f(θ_Q x_t; W_t)
```

TTT-Linear uses f_res(x) = Wx; TTT-MLP uses a two-layer MLP with 4x hidden width and GELU. Both wrap the inner model as f(x) = x + LN(f_res(x)). W_0 is a learnable outer-loop parameter ("learning W_0 significantly improves training stability"). Mini-batch TTT evaluates all b gradients of a mini-batch at the same base point, G_t = ∇ℓ(W_{t'}; x_t) with t' = t − mod(t, b), so the b gradient evaluations run in parallel; b = 16 was used, and smaller b gives lower perplexity at the cost of parallelism. The dual form avoids materializing per-token G_t and W_t. For a mini-batch with views X_K, X_V, X_Q (TTT-Linear, LN and residual omitted):

```
W_b = W_0 − 2η (W_0 X_K − X_V) X_Kᵀ
Z   = W_0 X_Q − 2η (W_0 X_K − X_V) mask(X_Kᵀ X_Q)
```

which is matmul-only. Inner base learning rates were 1.0 (TTT-Linear) and 0.1 (TTT-MLP); the outer loop is AdamW on a Mamba-style backbone with temporal convolutions. At 125M to 1.3B, both TTT variants keep improving with context where Mamba plateaus after 16K, and TTT-Linear is faster than the Transformer at 8K context.

What it changed. Reframed the RNN state as a trainable model and introduced the inner/outer-loop recipe that everything later reuses: learned W_0, learned per-token learning rate, LN plus residual on the inner model, and the dual form for chunkwise parallelism. TTT-E2E later refers to this reconstruction-loss family as TTT-KVB (key-value binding).

## 2. LaCT: Test-Time Training Done Right (2025)

Tianyuan Zhang, Sai Bi, Yicong Hong, Kai Zhang, Fujun Luan, Songlin Yang, Kalyan Sunkavalli, William T. Freeman, Hao Tan. arXiv:2505.23884, 2025-05-29.

Mechanism. Diagnoses why small-chunk TTT is slow: with fast-weight size h and chunk b, arithmetic intensity is bounded by r ≤ min(h/2, b) FLOPs per byte, so h = 64, b = 16 gives about 8 versus about 290 for an H100 and under 5% utilization. LaCT updates once per very large chunk (2K to 1M tokens), which is matmul-bound, reaches about 70% utilization on A100 in plain PyTorch, and lets the nonlinear state grow to about 40% of parameters. The fast weight is a SwiGLU-MLP f_W(x) = W_2[SiLU(W_1 x) ∘ (W_3 x)]. The inner loss is the negative dot product L_i = −f_W(k_i)ᵀ v_i with L2-normalized q and k; the per-token learning rate is η_i = softplus(Linear(x_i) + bias). The update is

```
g = Σ_{i∈chunk} η_i ∇L_i           (optionally M ← βM + g)
W ← L2norm(W − Muon(g))            Muon(g) ≈ UVᵀ for g = USVᵀ (Newton-Schulz; the reference code uses a 5-iteration routine)
o_i = f_W(q_i)
```

L2 normalization of W along the input dimension replaces weight decay. "Apply then update" (a shifted block-causal mask) is used for language modeling so tokens never see their own chunk through the fast weights; a window-attention layer supplies intra-chunk context. Results: 760M and 3B language models at 32K (chunks 2048 and 4096) competitive with DeltaNet and GLA, novel-view synthesis up to 1M tokens, and a 14B autoregressive video diffusion model at 56K tokens.

What it changed. Made large nonlinear fast weights practical: chunk size became a hardware decision rather than a quality decision; Muon with normalized weights became the stable inner optimizer; and the intra-chunk versus inter-chunk split became the standard argument for hybrid designs.

## 3. The Google line: Titans, MIRAS, Atlas, Nested Learning, and 2026 successors

### 3.1 Titans: Learning to Memorize at Test Time
Ali Behrouz, Peilin Zhong, Vahab Mirrokni. arXiv:2501.00663, 2024-12-31.

A deep MLP memory M (L_M ≥ 2 layers) is trained online on ℓ(M_{t−1}; x_t) = ||M_{t−1}(k_t) − v_t||². "Surprise" is the gradient; it is accumulated with momentum and a data-dependent forget gate:

```
S_t = η_t S_{t−1} − θ_t ∇ℓ(M_{t−1}; x_t)
M_t = (1 − α_t) M_{t−1} + S_t
y_t = M*(q_t)                        (retrieval without update)
```

Three integrations: MAC (retrieved memory concatenated as context for attention), MAG (memory output gates a sliding-window attention branch), MAL (memory as a layer before attention), plus learnable persistent-memory tokens. Training is chunkwise with matmuls and sums. Reports context beyond 2M tokens and BABILong gains. What it changed: forgetting and momentum became first-class parts of the inner loop, and a nonlinear memory was paired with attention rather than replacing it.

### 3.2 MIRAS: It's All Connected
Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni. arXiv:2504.13173, 2025-04-17.

Casts every sequence layer as online regularized optimization of an associative memory:

```
W_t = argmin_W  ℓ̃_t(W; k_t, v_t) + Ret_t(W, W_{t−1})
```

with four design choices: memory architecture, attentional bias (the inner loss), retention gate (forgetting as ℓ2 retention regularization), and the memory learning algorithm. Transformers are the dot-product-bias, nonparametric, no-retention case. New variants: Moneta (ℓ_p bias with ℓ_q retention: A_t = α_t A_{t−1} − η_t ∇ℓ_p(W_{t−1}; k_t, v_t), W_t = A_t / ||A_t||_q^{q−2}), Yaad (Huber loss, robust to outlier tokens), Memora (KL retention: W_t = Softmax(α_t log W_{t−1} − η_t ∇ℓ_2(W_{t−1}; k_t, v_t))). What it changed: the loss and the regularizer became design knobs alongside the optimizer.

### 3.3 ATLAS: Learning to Optimally Memorize the Context at Test Time
Ali Behrouz, Zeman Li, Praneeth Kacham, Majid Daliri, Yuan Deng, Peilin Zhong, Meisam Razaviyayn, Vahab Mirrokni. arXiv:2505.23735, 2025-05-29.

Replaces the online (last-token) objective with the Omega rule, a sliding-window objective over the last c tokens, optimized with Muon:

```
min_M Σ_{i=t−c+1}^{t} γ_i^{(t)} ||M(φ_p(k_i)) − v_i||²
M_t = α_t M_{t−1} − η_t NewtonSchulz_k(S_t)
```

where φ_p is a polynomial feature map on keys (capacity O(d_k^p) associations) and S_t is the momentum-accumulated gradient. DeepTransformers and Dot generalize softmax attention with deep memories and exponential feature maps. Reports +80% accuracy over Titans at 10M-token BABILong. What it changed: "locally optimal" memory (a windowed objective and an orthogonalized step) rather than one SGD step per token.

### 3.4 Nested Learning: The Illusion of Deep Learning Architectures (HOPE)
Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni. NeurIPS 2025; arXiv:2512.24695, 2025-12-31; Google Research blog 2025-11-07 (https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/).

Models are nested optimization levels ordered by update frequency. Momentum itself is an associative memory of gradients, m_{t+1} = α_{t+1} m_t − η_{t+1} ∇L(W_t; x_{t+1}), which licenses "deep optimizers" (replace dot-product similarity with ℓ2 regression, add preconditioning). The Continuum Memory System is a chain of MLP blocks where level ℓ updates every C^{(ℓ)} steps with its own η_ℓ. HOPE is self-modifying Titans (the memory generates its own key, value and query projections and learning rates) plus CMS. Tested at 340M, 760M and 1.3B against Transformer++, Titans, Samba and Gated DeltaNet on language modeling, continual learning, needle-in-haystack and BABILong. What it changed: multi-timescale memories and the optimizer-as-memory view. This is the closest existing work to "learned optimizers inside the sequence model".

### 3.5 2026 successors from the same group
- **Memory Caching: RNNs with Growing Memory.** Behrouz, Li, Deng, Zhong, Razaviyayn, Mirrokni. arXiv:2602.24281, 2026-02-27. Caches checkpoints of the recurrent memory so effective capacity grows with length; four variants with gated aggregation and sparse selection; competitive with Transformers on recall-heavy tasks.
- **Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories.** Behrouz, Hashemi, Javanmard, Mirrokni. arXiv:2606.03979, 2026-06-02 (OpenReview version from Sept 2025). Consolidates fragile fast-weight memory into slow weights via "Knowledge Seeding" (on-policy distillation into a larger network) and a "Dreaming" RL self-curriculum. The first explicit inner-to-outer consolidation loop from this line.
- **Proteus: Incremental Memory Activation for Long-Context Sequence Modeling.** Bayat, Behrouz, Mirrokni, Courville. arXiv:2608.16844, 2026-08-17. Schedules effective memory capacity to grow with context, applied to Titans, Hope-attention, SWLA and Comba; gains grow with context length.

No Google paper or blog claims a Titans or HOPE layer in a shipped Gemini model. The 2025-12-04 Titans+MIRAS blog (https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/) contains no deployment statements; third-party "roadmap" posts asserting Gemini integration are not sourced from Google.

## 4. End-to-end TTT and the 2026 "TTT for pretrained LLMs" wave

### 4.1 End-to-End Test-Time Training for Long Context (TTT-E2E)
Arnuv Tandon, Karan Dalal, Xinhao Li, Daniel Koceja, Marcel Rød, Sam Buchanan, Xiaolong Wang, Jure Leskovec, Sanmi Koyejo, Tatsunori Hashimoto, Carlos Guestrin, Jed McCaleb, Yejin Choi, Yu Sun. arXiv:2512.23675, 2025-12-29 (v2 2025-12-31). Code: https://github.com/test-time-training/e2e

The architecture is a plain Transformer with 8K sliding-window attention. At test time the model keeps doing next-token prediction on the context it reads, with plain SGD, a fixed scalar η, and mini-batches of b = 1K tokens:

```
ℓ_t(W) = CE(f(x_{<t}; W), x_t)
W_i = W_{i−1} − η (1/b) Σ_{t=(i−1)b+1}^{ib} ∇ℓ_t(W_{i−1})
```

Only the MLPs of the last quarter of blocks are updated, and those blocks get a second, static MLP as "safe" storage for pre-trained knowledge. W resets to W_0 at each document. The outer loop meta-trains W_0 through the inner loop:

```
L(W_0; X) = (1/T) Σ_{i=1}^{T/b} Σ_{t∈batch i} ℓ_t(W_{i−1})
```

which needs gradients of gradients; memory is handled by gradient checkpointing through time with all T/b steps unrolled. b = 8K "significantly hurts". At 3B parameters and 164B tokens, loss versus context tracks full attention where Mamba-2, Gated DeltaNet and TTT-KVB flatten; latency is constant in context and 2.7x faster than full attention at 128K; needle-in-a-haystack is much worse than full attention. What it changed: the inner loss is the task loss rather than a reconstruction proxy, and the meta-learned initialization is what makes small-η SGD on raw next-token loss work.

### 4.2 2026 follow-ups
- **In-Place Test-Time Training.** Guhao Feng, Shengjie Luo, Kai Hua, Ge Zhang, Di He, Wenhao Huang, Tianle Cai. ICLR 2026 oral; arXiv:2604.06169, 2026-04-07. Fast weights are the existing MLP down-projection W_down (W_up and W_gate frozen). The target is V̂ = Conv1D(X_0) W_target from token embeddings with loss −⟨·,·⟩_F, giving a chunked rank-b write W_down^{(i)} = W_down^{(i−1)} + η V̂_[i]ᵀ Z_[i] with Z the MLP activations; chunk 512 to 1024; a three-stage prefix sum makes it context-parallel. Retrofitted to Qwen3-4B-Base (20B tokens at 32K, then 15B at 128K).
- **Test-Time Training with KV Binding Is Secretly Linear Attention.** Junchen Liu, Sven Elflein, Or Litany, Zan Gojcic, Ruilong Li. ICML 2026; arXiv:2602.21204, 2026-02-24. For any fast weight with a linear bias-free last layer f(x) = φ(x; Θ)W, one gradient step gives o_t = φ_{t+1}(q_t)(W_0 + Σ_i φ_i(k_i)ᵀ g_i(k_i)), which is linear attention with learned kernels; momentum and per-token learning rates fold into the effective values. Removing weight normalization makes the recurrence associative and scan-parallel: up to 4.0x inference throughput at +0.87 perplexity. A caution against reading TTT-KVB as memorization.
- **Test-Time Training with Next-Token Prediction (TTT-NTP).** Xuan Ouyang, Zefan Cai, Junjie Hu. Findings of EMNLP 2026; arXiv:2606.21803, 2026-06-19. Drop-in fast weights for pretrained checkpoints whose target is the model's own next contextual hidden state; +2.9 to +4.1 on RULER 4K to 32K across Llama-3.1-8B, Mistral-7B and Qwen3-4B/0.6B.
- **Let's (not) just put things in Context.** Rachit Bansal et al. arXiv:2512.13898, 2025-12-15. **Self-Guided TTT.** Xinyu Zhu et al. arXiv:2607.09415, 2026-07-10. Whole-model gradient updates on the context beat thinking-token scaling at long context; S-TTT lets the model select the evidence spans to train on (up to 15% relative gain on LongBench-v2 and LongBench-Pro).
- **Beyond Perplexity.** Xiangchen Song et al. arXiv:2607.00368, 2026-07-01. Proxy-loss gains from one-step LoRA TTT on Qwen3 do not translate into behavioral recall.
- **REFINE.** Hee Seung Hwang, Xindi Wu, Sanghyuk Chun, Olga Russakovsky. arXiv:2602.16704, 2026-02-18. Trains LaCT-760M and DeltaNet-1.3B with GRPO on a next-sequence objective; beats next-token fine-tuning on needle-in-haystack and LongBench.
- From Yu Sun's group, the only 2026 entries are Learning to Discover at Test Time (arXiv:2601.16175, 2026-01-22, RL at test time rather than a TTT layer) and a robotics paper; no TTT-E2E successor had appeared by 2026-09-21 (author page: https://www.alphaxiv.org/@yu-sun).

## 5. Delta-rule, SSM and linear fast-weight family

### 5.1 Core 2024 to 2025 papers
- **Parallelizing Linear Transformers with the Delta Rule over Sequence Length (DeltaNet).** Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim. arXiv:2406.06484, 2024-06-10 (camera-ready v3 2025-01-15). S_t = S_{t−1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ is one SGD step on ½||S k_t − v_t||²; a Householder (WY) representation makes it chunk-parallel; 1.3B at 100B tokens beats Mamba and GLA.
- **Gated Delta Networks: Improving Mamba2 with Delta Rule.** Songlin Yang, Jan Kautz, Ali Hatamizadeh. ICLR 2025; arXiv:2412.06464, 2024-12-09. Adds a scalar decay: S_t = S_{t−1}(α_t (I − β_t k_t k_tᵀ)) + β_t v_t k_tᵀ. Gating erases, the delta rule edits. Basis of Qwen3-Next, Qwen3.5 and Qwen3.6.
- **Transformers are SSMs (Mamba-2 / SSD).** Tri Dao, Albert Gu. ICML 2024; arXiv:2405.21060, 2024-05-31. Scalar-decay diagonal SSM h_t = a_t h_{t−1} + B_t x_t viewed as a semiseparable matrix; 2 to 8x faster than Mamba. In the regression view it is exponentially weighted linear attention.
- **Longhorn: State Space Models are Amortized Online Learners.** Bo Liu, Rui Wang, Lemeng Wu, Yihao Feng, Peter Stone, Qiang Liu. arXiv:2407.14207, 2024-07-19. S_t = argmin_S ||S − S_{t−1}||_F² + ||S k_t − x_t||²_{diag β_t} with closed form S_t = (I − ε_t k_t k_tᵀ) S_{t−1} + ε_t k_t x_tᵀ, ε_t = β_t / (1 + β_t k_tᵀ k_t); a diagonal approximation gives a Mamba-style scan with no separate A. 1.8x data efficiency versus Mamba and 16x length extrapolation. An implicit (proximal) inner step instead of explicit SGD.
- **DeltaProduct.** Julien Siems, Timur Carstensen, Arber Zela, Frank Hutter, Massimiliano Pontil, Riccardo Grazzi. NeurIPS 2025; arXiv:2502.10297, 2025-02-14. n_h delta steps per token, A_t = Π_{j=1}^{n_h} (I − β_{t,j} k_{t,j} k_{t,j}ᵀ), optionally gated by g_t; products of Householders give rotations and therefore S_n state tracking; n_h = 2 already fixes length extrapolation at 1.3B.
- **RWKV-7 "Goose".** Bo Peng et al. (18 authors). arXiv:2503.14456, 2025-03-18. S_t = S_{t−1}(diag(w_t) − κ̂_tᵀ(a_t ⊙ κ̂_t)) + v_tᵀ k̃_t with vector decay w_t = exp(−e^{−0.5} σ(d_t)), vector in-context learning rate a_t ∈ (0,1), normalized removal key κ̂_t and a decoupled replacement key k̃_t = k_t ⊙ lerp(1, a_t, α). Recognizes all regular languages with constant layers; 2.9B model under Apache-2.0.
- **Lattice.** Mahdi Karami, Razvan Pascanu, Vahab Mirrokni. arXiv:2504.05646, 2025-04-08. Memory slots updated only with the component of the input orthogonal to their current state, derived as one gradient step on an online low-rank compression objective.
- **Test-time regression.** Ke Alexander Wang, Jiaxin Shi, Emily B. Fox. arXiv:2501.12352, 2025-01-21 (v3 2025-05-02). m_t = argmin_m ½ Σ_i γ_i^{(t)} ||v_i − m(k_i)||², y_t = m_t(q_t). Linear attention assumes KᵀK ≈ I; DeltaNet is streaming SGD; gated models use geometric γ; softmax is locally-constant kernel regression, exact only when ||k|| = ||q|| = 1 (justifying QK-norm); higher-order local-linear attention is proposed.
- **MesaNet: Sequence Modeling by Locally Optimal Test-Time Training.** Johannes von Oswald et al. (17 authors). ICLR 2026; arXiv:2506.05233, 2025-06-05 (v2 2026-06-03). Solves the regularized least squares exactly at every step: G_t = γ_t G_{t−1} + β_t v_t k_tᵀ, H_t = γ_t H_{t−1} + β_t k_t k_tᵀ, o_t = G_t (H_t + Λ)^{−1} q_t via conjugate gradient (30 steps in training, tolerance-based at inference), chunkwise-parallel, with context-dependent forgetting. Best perplexity among RNNs at 140M to 1B on 50B tokens, but "transformers >> MesaNet" on in-context recall.
- **Log-Linear Attention.** Han Guo, Songlin Yang, Tarushii Goel, Eric P. Xing, Tri Dao, Yoon Kim. ICLR 2026; arXiv:2506.04761, 2025-06-05. Replaces one state with O(log T) states over a Fenwick-tree partition of the past; instantiated on Mamba-2 and Gated DeltaNet.
- **Mamba-3: Improved Sequence Modeling using State Space Principles.** Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao, Albert Gu. ICLR 2026 oral; arXiv:2603.15569, 2026-03-16. Exponential-trapezoidal discretization h_t = α_t h_{t−1} + β_t B_{t−1} x_{t−1} + γ_t B_t x_t with α_t = e^{Δ_t A_t}, β_t = (1 − λ_t) Δ_t e^{Δ_t A_t}, γ_t = λ_t Δ_t (makes the short conv optional); data-dependent rotary (complex) states that solve parity (100% versus 0.9% for Mamba-2); MIMO rank-R updates that raise arithmetic intensity without growing the state. +1.8 points over Gated DeltaNet at 1.5B. No delta rule: this is the "SSM principles" branch.

### 5.2 2026 delta-rule refinements (better inner-loop optimizers for linear fast weights)

| Paper | arXiv, date | What it adds to S_t = S_{t−1}(α(I − βkkᵀ)) + βvkᵀ |
|---|---|---|
| Kimi Linear / KDA (Kimi Team) | 2510.26692, 2025-10-30 | Channel-wise decay: S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ; 3:1 KDA:MLA; 48B-A3B on 5.7T tokens; 6x decode at 1M |
| FG²-GDN (Pingwei Sun et al.) | 2604.19021, 2026-04-21 | β becomes a channel-wise vector ("SGD to Adagrad/Adam"); FG²-GDN+ decouples key and value scaling |
| Preconditioned DeltaNet (Tumma, Loo, Rus) | 2604.21100, 2026-04-22 | Diagonal curvature preconditioner on the implicit least-squares loss; variants for DeltaNet, GDN, KDA |
| OSDN (Chenyu Zhou et al.) | 2605.13473, 2026-05-13 | Diagonal preconditioner updated online by hypergradient with adaptive forgetting; +32% recall at 340M |
| MDN (Yulong Huang et al.) | 2605.05838, 2026-05-07 | Stepwise momentum inside the delta rule, chunk-parallel; complex-conjugate eigenvalue analysis for gate design |
| Kaczmarz Linear Attention (Zou, Ren, Liu) | 2605.08587, 2026-05-09 | β_t derived from a Kaczmarz projection (key-norm normalized) instead of learned; 8.09 versus 8.50 ppl at 0.4B |
| Gated DeltaNet-2 (Hatamizadeh, Choi, Kautz) | 2605.22791, 2026-05-21 | Separate channel-wise erase gate b_t and write gate w_t; beats GDN, KDA and Mamba-3 at 1.3B on 100B tokens, especially multi-key RULER |
| Q-Delta (Park, Kim, Park; ICML 2026) | 2606.08804, 2026-06-07 | Query-conditioned prediction error enters the state update, not only the readout |
| PRISM (Jie Jiang et al.) | 2602.10796, 2026-02-11 | Replaces the serial inner loop with a two-stage proxy giving rank-L accumulation; 174x throughput versus optimization-based TTT |
| M²RNN (Mishra, Tan, Stoica, Gonzalez, Dao) | 2603.14360, 2026-03-15 | Non-linear matrix-state RNN beyond TC⁰, tensor-core friendly |

Related 2026 work: Adaptive Memory Decay for Log-Linear Attention (arXiv:2605.06946, 2026-05-07) and Key-Value Means (arXiv:2605.09877, 2026-05-11). The pattern: every knob of the inner optimizer (step size, per-coordinate scaling, momentum, preconditioning, erase/write decoupling, query-aware error) has now been tried on the linear delta rule, mirroring the SGD to Adam to Muon progression of outer-loop optimizers.

## 6. Applications beyond text

- **One-Minute Video Generation with Test-Time Training.** Karan Dalal, Daniel Koceja, Gashon Hussein, Jiarui Xu, Yue Zhao, Youjin Song, Shihao Han, Ka Chun Cheung, Jan Kautz, Carlos Guestrin, Tatsunori Hashimoto, Sanmi Koyejo, Yejin Choi, Yu Sun, Xiaolong Wang. CVPR 2025; arXiv:2504.05298, 2025-04-07. TTT-MLP layers (2-layer MLP hidden state) inserted into pre-trained CogVideo-X 5B, gated, with 3-second segments processed bidirectionally inside a segment and causally across segments, plus an on-chip tensor-parallel kernel. +34 Elo over Mamba-2, Gated DeltaNet and sliding-window baselines on Tom and Jerry storyboards.
- **tttLRM.** Chen Wang, Hao Tan, et al. CVPR 2026; arXiv:2602.20160, 2026-02-23. LaCT-style fast weights as an implicit 3D representation decodable to Gaussian splats, with an online streaming variant.
- **Fast Spatial Memory with Elastic TTT.** Ziqiao Ma, Xueyang Yu, Haoyu Zhen, Yuncong Yang, Joyce Chai, Chuang Gan. arXiv:2604.07350, 2026-04-08. Stabilizes LaCT with a Fisher-weighted elastic prior toward an EMA anchor of past fast weights so many small chunks can be processed without forgetting. Directly relevant to rollback-style safety.
- **Spatial-TTT.** Fangfu Liu et al. arXiv:2603.12255, 2026-03-12. A 3:1 TTT:attention hybrid with large-chunk updates for streaming spatial video.
- **LongVU-TTT.** Mahmoud Ahmed et al. arXiv:2608.25729, 2026-08-26. **Forget, Anticipate and Adapt.** Rajat Modi et al. ECCV 2026; arXiv:2606.26515, 2026-06-25. Causal fast-weight resamplers and surprise-driven window sizing for multi-hour video.

## 7. Production adoption as of 2026-09-21

- **Gated DeltaNet is in production.** Qwen3-Next-80B-A3B (Sept 2025; 3:1 Gated DeltaNet to Gated Attention; https://vllm.ai/blog/2025-09-11-qwen3-next), Qwen3.5 (Feb 2026, same 3:1 hybrid; https://huggingface.co/blog/mlabonne/qwen35) and Qwen3.6. Kimi Linear (KDA, 3:1 with MLA, Oct 2025) is open-weight. NVIDIA Nemotron 3 Nano, Super and Ultra (arXiv:2512.20856, 2025-12-24) are Mamba-2 plus attention plus MoE hybrids with up to 1M context. Ling 2.5 and 3.0 use Lightning Attention hybrids (https://sebastianraschka.com/llm-architecture-gallery/hybrid-attention/).
- **TTT and neural-memory layers are not in any shipped frontier model.** In-Place TTT and TTT-NTP retrofit Qwen3 and Llama checkpoints in research settings; TTT-E2E code and checkpoints (125M to 3B) are public. No Google, OpenAI, Anthropic or Alibaba model card claims Titans, HOPE or TTT layers.

## 8. Design lessons and hyperparameters

1. **Inner objective.** Reconstruction or KV binding (||f_W(k) − v||² or −f_W(k)ᵀv) is cheap, layer-local and parallel, but with a linear last layer it collapses to learned linear attention (Liu et al. 2026). The task loss (next-token CE in TTT-E2E; NTP-aligned targets in In-Place TTT and TTT-NTP) is what tracks full-attention scaling at 128K. In-Place TTT's compromise is an NTP-aligned linear target of the next token's embedding with a dot-product loss.
2. **Inner optimizer.** Plain SGD with a fixed scalar η works for TTT-E2E because W_0 is meta-learned. For nonlinear fast weights without E2E meta-training, LaCT and Atlas both use Muon (Newton-Schulz orthogonalized gradient) with momentum; Muon makes η encode only the relative importance of tokens within a chunk. For linear fast weights, the 2026 papers show that diagonal preconditioning, channel-wise β and Kaczmarz step sizes each buy recall.
3. **Chunk size.** Quality prefers small chunks (TTT: b = 16 is best in perplexity; TTT-E2E: b = 1K, and 8K "significantly hurts"); hardware prefers large chunks (LaCT: intensity r ≤ min(h/2, b), 2K to 4K for language modeling). LaCT's "small-chunk TTT is bad" is a hardware-efficiency claim, not a quality claim, and does not bind for a toy model on one GPU.
4. **Learning-rate gating.** Every nonlinear-memory paper uses an input-dependent learning rate: η(x) = η_base σ(θ_lr x) with η_base 1.0 linear and 0.1 MLP (TTT), softplus(Linear(x) + bias) (LaCT), θ_t (Titans), vector a_t (RWKV-7), channel-wise β (FG²-GDN, KDA).
5. **Forgetting.** Scalar decay α_t (Gated DeltaNet; Titans uses (1 − α_t)), channel-wise decay (KDA, RWKV-7, GDN-2), or replace decay with L2 normalization of W rows (LaCT). MIRAS reads forgetting as ℓ2 retention regularization and Memora uses KL. Elastic TTT adds a Fisher-weighted prior toward an anchor.
6. **Dual form.** With a fixed base point per mini-batch, outputs are Z = W_0 X_Q − 2η (W_0 X_K − X_V) mask(X_Kᵀ X_Q) and no per-token W_t is materialized; Titans, Atlas and LaCT generalize this to "update once per chunk, apply with matmuls".
7. **Stability.** Learn W_0; LN plus residual around the fast-weight output (TTT); L2-normalize q and k (LaCT); normalize or gate W; keep a static parallel path for pre-trained knowledge (TTT-E2E's second MLP, In-Place TTT's frozen W_up and W_gate); reset W at document boundaries; checkpoint through time when unrolling many inner steps.

## 9. Design implications for a toy TTT x SSM model in PyTorch (≤5M params, one GPU, under 1 hour)

**Formulation to pick.** A Gated-DeltaNet or Mamba-2 style diagonal selective SSM supplies intra-chunk and short-range context; a LaCT-style chunked nonlinear fast-weight memory supplies cross-chunk in-context learning; the whole thing is meta-trained end-to-end through the inner loop (TTT-E2E style outer loop) but with a KV-binding inner loss (TTT and LaCT style), because a full-model next-token inner loss is too expensive at this budget. Reasons: the SSM scan is cheap and already exists in this repo; apply-then-update chunking is causal by construction and needs no kernel; the hybrid mirrors LaCT (window attention plus TTT) and Titans MAG, where the nonlinear memory carries the long-range part and a local mixer the rest; and meta-learning W_0, θ_lr and the forget gate is what made small-η SGD stable in TTT and TTT-E2E.

**Layer (per block, input x_t ∈ ℝ^d, d = 128, chunk size b = 64 to 128).**

SSM branch (context):

```
Δ_t = softplus(W_Δ x_t),   a_t = exp(−Δ_t ⊙ exp(A)),   B_t = W_B x_t,   C_t = W_C x_t
h_t = a_t ⊙ h_{t−1} + Δ_t ⊙ (B_t ⊗ x_t)
c_t = LN(C_tᵀ h_t + D ⊙ x_t)
```

TTT branch (memory), fast weights W = {W_1 ∈ ℝ^{2d×d}, W_2 ∈ ℝ^{d×2d}}, f_W(u) = W_2 GELU(W_1 u):

```
k_t = norm(W_K c_t),   v_t = W_V c_t,   q_t = norm(W_Q c_t)
η_t = η_base · σ(w_lrᵀ c_t + b_lr)                    (η_base ≈ 0.1)
α_j = σ(w_αᵀ mean_{t∈C_j} c_t + b_α)                  (per-chunk forget gate)
L_j(W) = Σ_{t∈C_j} η_t ||f_W(k_t) − v_t||²
G_j = ∇_W L_j(W_{j−1})                                 (torch.func.grad, or autograd.grad with create_graph=True)
W_j = (1 − α_j) W_{j−1} − G_j                          (Muon/LaCT variant: W_j = rownorm(W_{j−1} − NS5(G_j)))
o_t = c_t + LN(f_{W_{j−1}}(q_t))   for t ∈ C_j         (apply-then-update: no token reads its own chunk)
```

Combine (MAG style), then a standard MLP block:

```
y_t = x_t + W_O (σ(W_g x_t) ⊙ o_t)
```

Because o_t only sees chunks before j through W, the SSM branch is the only path for within-chunk information; this is the same intra/inter-chunk division of labor as LaCT.

**Outer loop (meta-training).** Parameters Θ = {W_Δ, A, W_B, W_C, D, W_K, W_V, W_Q, W_O, W_g, w_lr, b_lr, w_α, b_α, W_0 = (W_1^0, W_2^0), LNs, embeddings, head}. Reset W to W_0 at each sequence. Forward the full sequence, running T/b inner updates; the outer loss is ordinary next-token cross-entropy,

```
L_outer(Θ) = (1/T) Σ_t CE(head(y_t), x_{t+1})
```

backpropagated through every inner step (second-order gradients through G_j). With T = 2048 and b = 128 there are 16 inner steps per sequence, so full unrolling without checkpointing is fine in a 5M model. Train with AdamW (lr 3e-3, cosine), batch 16 to 32, about 3K to 5K steps. Ablations worth running: b ∈ {32, 64, 128, 256}; η_base ∈ {0.03, 0.1, 0.3}; SGD versus a 5-step Newton-Schulz inner step; forget gate versus L2-normalized W; KV-binding loss versus an E2E inner loss on the chunk's own next-token CE (the TTT-E2E variant: replace L_j by Σ_{t∈C_j} CE(head(y_t), x_{t+1}) computed with W_{j−1}, which costs one extra block forward per chunk).

**Parameter budget (d = 128, 4 blocks, 8K vocab).** Embedding with tied head about 1.0M; per block: SSM about 70K, TTT projections 4d² about 66K, fast-weight initialization W_0 about 66K, MLP about 130K, gates and LNs about 20K, so about 350K per block and about 2.4M total.

**Why this and not the alternatives.** A pure Gated DeltaNet is simpler and is what production uses, but its inner loop is closed-form and there is nothing for an inner-loop safety monitor to observe beyond β_t and α_t. A pure TTT-E2E is the strongest long-context formulation but needs a full block forward per inner step plus checkpointing through time, which does not fit a one-hour budget at interesting chunk sizes. MesaNet's exact solve is elegant but its conjugate-gradient loop is the wrong place to spend a toy budget. The chosen hybrid keeps the quantities this repository already monitors, the inner gradient norm ||G_j||, the update norm ||W_j − W_{j−1}||, and their alignment with a canary gradient, as per-chunk signals; the forget gate α_j and the Fisher-weighted anchor from Fast Spatial Memory are principled versions of the pre-update gate and post-update rollback.

## Sources

- https://arxiv.org/abs/2407.04620 (TTT layers) and https://arxiv.org/html/2407.04620v2
- https://arxiv.org/abs/2505.23884 (LaCT) and https://arxiv.org/html/2505.23884v1
- https://arxiv.org/abs/2501.00663 (Titans), https://arxiv.org/abs/2504.13173 (MIRAS), https://arxiv.org/abs/2505.23735 (Atlas), https://arxiv.org/abs/2512.24695 (Nested Learning)
- https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/ and https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/
- https://arxiv.org/abs/2602.24281, https://arxiv.org/abs/2606.03979, https://arxiv.org/abs/2608.16844, https://arxiv.org/abs/2606.23670 (Behrouz 2026); https://www.alphaxiv.org/@ali-behrouz
- https://arxiv.org/abs/2512.23675 (TTT-E2E), https://github.com/test-time-training/e2e, https://www.alphaxiv.org/@yu-sun
- https://arxiv.org/abs/2604.06169, https://arxiv.org/abs/2602.21204, https://arxiv.org/abs/2606.21803, https://arxiv.org/abs/2512.13898, https://arxiv.org/abs/2607.09415, https://arxiv.org/abs/2607.00368, https://arxiv.org/abs/2602.16704
- https://arxiv.org/abs/2406.06484, https://arxiv.org/abs/2412.06464, https://arxiv.org/abs/2405.21060, https://arxiv.org/abs/2407.14207, https://arxiv.org/abs/2502.10297, https://arxiv.org/abs/2503.14456, https://arxiv.org/abs/2504.05646, https://arxiv.org/abs/2501.12352, https://arxiv.org/abs/2506.05233, https://arxiv.org/abs/2506.04761, https://arxiv.org/abs/2603.15569
- https://arxiv.org/abs/2510.26692, https://arxiv.org/abs/2604.19021, https://arxiv.org/abs/2604.21100, https://arxiv.org/abs/2605.13473, https://arxiv.org/abs/2605.05838, https://arxiv.org/abs/2605.08587, https://arxiv.org/abs/2605.22791, https://arxiv.org/abs/2606.08804, https://arxiv.org/abs/2602.10796, https://arxiv.org/abs/2603.14360, https://arxiv.org/abs/2605.06946, https://arxiv.org/abs/2605.09877
- https://arxiv.org/abs/2504.05298, https://arxiv.org/abs/2602.20160, https://arxiv.org/abs/2604.07350, https://arxiv.org/abs/2603.12255, https://arxiv.org/abs/2608.25729, https://arxiv.org/abs/2606.26515
- https://vllm.ai/blog/2025-09-11-qwen3-next, https://huggingface.co/blog/mlabonne/qwen35, https://arxiv.org/abs/2512.20856, https://sebastianraschka.com/llm-architecture-gallery/hybrid-attention/
- Survey: https://arxiv.org/abs/2508.09834 (Speed Always Wins, 2025-08-13)
