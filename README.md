# AegisStep-28M

A ~28M-parameter **Process Reward Model (PRM)** built from scratch in PyTorch. It reads a math reasoning chain and scores every intermediate step as correct or incorrect.

Every component is implemented by hand: sparse Mixture-of-Experts routing, Grouped-Query Attention, rotary embeddings, SwiGLU experts and RMSNorm. The model is trained from random initialization on [Math-Shepherd](https://huggingface.co/datasets/peiyi9979/Math-Shepherd).

**Weights and model card:** https://huggingface.co/saqiibb/AegisStep-28M

> **Status:** research and educational baseline. It is not a reliable verifier. See [Limitations](#limitations).

## Results

Evaluated on **10,000 held-out reasoning chains**. Splits are made by question, so no question appears in more than one split.

| Metric | Value |
|---|---|
| **Step-level AUROC** | **0.773** |
| Balanced accuracy @ 0.5 | 69.99% |
| Accuracy @ 0.5 | 69.97% |
| Majority-class baseline accuracy | 50.54% |
| Full-chain accuracy (every step right) | 39.97% |

At the validation-tuned threshold of 0.30: accuracy 67.87%, precision 63.70%, recall 84.67%, F1 72.70%. A model that always predicts "correct" scores 67.14% F1 on this test set, so **AUROC and balanced accuracy are the more informative numbers**. Full-chain accuracy is strict, since one wrong step fails the whole chain.

## Architecture

| | |
|---|---|
| Parameters | ~28M (about half are the token embedding) |
| Layers | 6 |
| Hidden size | 288 |
| Attention | Grouped-Query Attention, 8 query heads, 2 KV heads |
| Feed-forward | Sparse MoE: 4 SwiGLU experts per layer, top-2 routing, `d_ff` = 576 |
| Load balancing | Auxiliary router loss (coefficient 1e-4) |
| Positions | RoPE, max sequence length 512 |
| Normalization | RMSNorm (pre-norm) |
| Dropout | 0.1 |
| Tokenizer | GPT-2 BPE plus `<step>`, `<prm_start>`, `<prm_end>` |

Input format:

```
<prm_start> {question} Step 1: ... <step> Step 2: ... <step> ... <prm_end>
```

A linear reward head reads the hidden state at each `<step>` token and outputs one logit. A sigmoid turns it into P(step is correct). Attention is causal, so each step's score depends on the question and the steps before it.

## Training

- **Data:** 150,000 Math-Shepherd chains (about 35k unique questions), shuffled with seed 42
- **Splits:** the question text is hashed into 90% train / 5% val / 5% test, so all solutions to one question stay in the same split. Validation and test use 10,000 chains each.
- **Sequence length:** 512 tokens, with dynamic per-batch padding
- **Optimizer:** AdamW, peak LR 3e-4, weight decay 0.1, one-cycle schedule (5% warmup, cosine decay), gradient clipping at 1.0, mixed precision
- **Batch size and epochs:** 32 and 3, keeping the checkpoint with the best validation loss (epoch 2)
- **Loss:** binary cross-entropy on `<step>` positions plus the MoE load-balancing loss
- **Hardware:** a single Kaggle GPU, about 30 minutes per epoch

## How to run

Everything (model code, data pipeline, training, evaluation) is in a single notebook: **`aegisstep_training.ipynb`**.

1. Open it on [Kaggle](https://www.kaggle.com) or Colab with a GPU enabled.
2. Run the cells from top to bottom. The first cell installs the required packages (`datasets`, `transformers`, `scikit-learn`, `huggingface_hub`).
3. Training takes about 30 minutes per epoch on a single Kaggle GPU (3 epochs).

The trained weights, tokenizer and config are on Hugging Face: https://huggingface.co/saqiibb/AegisStep-28M

## Repository contents

| File | Description |
|---|---|
| `aegisstep_training.ipynb` | Model architecture, data pipeline, training and evaluation |
| `README.md` |  |

## Lessons learned

Evaluation turned out to be harder than the model:

1. **Label parsing bug.** In Math-Shepherd, the step marker in the input becomes `+` or `-` in the label string. Splitting the label on whitespace produced almost all zeros, and the model reached 99.8% accuracy by predicting "incorrect" every time.
2. **Unshuffled slices.** The dataset is not shuffled, so contiguous slices gave almost all-positive test sets and inflated F1 to about 90%. Hash-based question-level splits with a shuffle fixed this.
3. **Truncation.** Half of the chains were longer than 256 tokens, so later steps (where errors tend to appear) were cut off. Moving to 512 tokens raised AUROC from 0.745 to 0.773.

## Limitations

- **Small and from scratch.** The model has never seen general language data. Fine-tuning a pretrained backbone would very likely score higher.
- **Noisy labels.** Math-Shepherd labels come from automatic Monte Carlo rollouts, not human annotation.
- **Truncation.** Chains longer than 512 tokens are cut, and later steps are not scored.
- **Domain.** Trained on math word problems only.
- **Overfitting.** Validation loss stopped improving after epoch 2.

## Possible next steps

- Fine-tune a small pretrained backbone (for example SmolLM or Qwen2.5-0.5B) with the same step head
- Train on the full ~440k Math-Shepherd chains
- Longer context to avoid truncation

## Citation

If you use the training data, please cite Math-Shepherd:

```bibtex
@article{wang2023math,
  title={Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations},
  author={Wang, Peiyi and Li, Lei and Shao, Zhihong and Xu, Runxin and Dai, Damai and Li, Yifei and Chen, Deli and Wu, Y. and Sui, Zhifang},
  journal={arXiv preprint arXiv:2312.08935},
  year={2023}
}
```

## License

Apache 2.0
