# Studying “Attention Is All You Need” with Mini-GPT

Use this repository as a scaffolding to internalize the 2017 Transformer paper. The code implements every decoder-side building block described by Vaswani et al., so reading and running it while keeping the paper open provides a concrete learning loop.

## 1. Quick Orientation
- **Scope match**: the repo implements the decoder stack (GPT-style). The paper also defines an encoder, but Sections 3.2–3.5—attention, multi-head projections, feed-forward towers, residual + normalization, positional encoding—are shared, so you can grasp most of the architecture here.
- **Key files**:
  - `apps/mini_gpt.py` — core model, mirrors the paper’s math.
  - `apps/train.py` — end-to-end workflow (tokenize → train → sample).
  - `apps/data_utils.py` — tokenizer helpers; good for understanding input preprocessing.

Keep the paper’s Figure 1 handy: every numbered box has a direct analogue in `mini_gpt.py`.

## 2. Preparation Checklist
1. Read the abstract and Section 1 once; note the claims about parallelization and sequence modeling.
2. Skim Section 2 (Background) to refresh attention terminology.
3. Install dependencies (`python3`, `numpy`) if needed: `pip install numpy`.

## 3. Layer-by-Layer Walkthrough

Follow the order of the forward pass in `MiniGPT.forward`.

### Step 3.1 — Token + Positional Embeddings
- Paper reference: Section 3.4–3.5.
- Code anchor: `apps/mini_gpt.py#L615` (token embeddings) and `apps/mini_gpt.py#L637` (positional encoding via `create_positional_encoding`).
- Exercise: derive the sine/cosine formula from the paper and verify that `create_positional_encoding` matches it. Plot a few positions/indices to visualize periodicity.

### Step 3.2 — Causal Masking
- Paper reference: Section 3.1 (decoder masking).
- Code anchor: `create_causal_mask` in `apps/mini_gpt.py#L166`.
- Exercise: print the mask for a short sequence to observe the lower-triangular structure; relate it to Figure 1’s masked self-attention block.

### Step 3.3 — Scaled Dot-Product Attention
- Paper reference: Equation (1) in Section 3.2.1.
- Code anchor: `scaled_dot_product_attention` at `apps/mini_gpt.py#L206`.
- Exercises:
  1. Walk through the matrix multiplications with a toy example (3-token sequence, dₖ = 2).
  2. Comment out the `1/√dₖ` scaling, rerun training for a few iterations, and note the instability described in the paper.

### Step 3.4 — Multi-Head Attention
- Paper reference: Section 3.2.2.
- Code anchor: `MultiHeadAttention.forward` (`apps/mini_gpt.py#L380`).
- Exercises:
  1. Track tensor shapes through `split_heads` → attention loop → `combine_heads`.
  2. Compare head dimension `d_model/num_heads` with the paper’s default (512/8 = 64) and experiment with different `num_heads` arguments when instantiating `MiniGPT`.

### Step 3.5 — Position-Wise Feed-Forward
- Paper reference: Section 3.3.
- Code anchor: `PositionWiseFeedForward.forward` (`apps/mini_gpt.py#L474`).
- Exercise: manually expand `FFN(x) = max(0, xW₁ + b₁)W₂ + b₂` and check how ReLU (not GELU) reflects the original paper.

### Step 3.6 — Residual Paths + LayerNorm
- Paper reference: Section 3.1.
- Code anchor: `TransformerBlock.forward` (`apps/mini_gpt.py#L535`).
- Exercises:
  1. Identify the order: attention → residual → LayerNorm, then FFN → residual → LayerNorm.
  2. Contrast with the “Pre-LN” variants used in later literature; understand why the original paper used post-norm.

### Step 3.7 — Output Projection
- Paper reference: Section 3.1 (final linear layer + softmax).
- Code anchor: final projection in `MiniGPT.forward` (`apps/mini_gpt.py#L669`).
- Exercise: inspect `compute_loss` (`apps/mini_gpt.py#L677`) to see cross-entropy computation and link back to Section 5.1’s training objective.

## 4. Running the Pipeline

1. **Tokenize & train** (Section 5.3 inspiration):
   ```bash
   python apps/train.py \
     --data data/sample.txt \
     --model_dir models/paper_walkthrough \
     --epochs 5 \
     --max_seq_length 64
   ```
   - Watch console output to connect loss curves with Section 5.4’s optimization discussion (note the simplifications: SGD on embeddings only, no Adam or warm-up).

2. **Generate text**:
   ```bash
   python apps/generate.py \
     --model_dir models/paper_walkthrough \
     --prompt "attention is" \
     --max_new_tokens 20
   ```
   - Observe how causal masking enforces left-to-right generation, echoing Section 3.1’s autoregressive decoder.

3. **Inspect artifacts** (`models/paper_walkthrough/`):
   - `model.pkl`: saved parameters (compare with Section 5.4’s weight matrices).
   - `tokenizer.pkl`: vocabulary, mirroring the paper’s use of token embeddings.

## 5. Mapping Cheat Sheet

| Paper Section | Concept | Repository Entry |
| --- | --- | --- |
| §3.2.1 | Scaled Dot-Product Attention | `apps/mini_gpt.py#L206` |
| §3.2.2 | Multi-Head Attention | `apps/mini_gpt.py#L298` |
| §3.3 | Position-wise FFN | `apps/mini_gpt.py#L436` |
| §3.4 | Embeddings Scaling | `apps/mini_gpt.py#L615` |
| §3.5 | Positional Encoding | `apps/mini_gpt.py#L120` |
| §3.1 | Residual + LayerNorm | `apps/mini_gpt.py#L535` |
| §5.3–5.4 | Training Setup | `apps/train.py#L55`, `apps/mini_gpt.py#L702` |

Use the table as a cross-reference while reading the corresponding subsections.

## 6. Suggested Paper Reading Order (Paired with Code)
1. **Abstract & Section 1** — motivation → run `apps/train.py` to see the goal in action.
2. **Section 3.2 (Self-Attention)** — pause to trace `scaled_dot_product_attention` and `MultiHeadAttention`.
3. **Section 3.3 (FFN)** — inspect the feed-forward class.
4. **Section 3.5 (Positional Encoding)** — reproduce the sinusoidal curves with NumPy.
5. **Section 5 (Training)** — compare the paper’s optimizer choices to the simplified training loop here; note omissions like Adam, label smoothing, and batched training.
6. **Section 6 (Results)** — reflect on performance gaps; consider what extra components (encoder, cross-attention) would be needed for translation tasks.

## 7. Extension Ideas
- **Recreate paper experiments**: add an encoder stack and cross-attention to mirror the full model from Figure 1.
- **Implement learning-rate warm-up and Adam**: replicate the optimization schedule in Section 5.3.
- **Introduce label smoothing** (Section 5.3) in `compute_loss` to observe its regularization effect.
- **Batch training**: replace the current single-sequence loop with batched matrix operations to connect with Section 3’s efficiency claims.

---

By alternating between reading specific paper sections and dissecting the companion code, you get both the theory (equations, ablations) and practice (NumPy tensors, actual execution). Treat each forward-pass component as a checkpoint: read the paragraph, find the matching implementation, run a quick experiment, and move on.
