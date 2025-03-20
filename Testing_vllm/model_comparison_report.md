# Model Comparison Report
*Generated on: 2025-03-17 22:17:42*

## Overview
This report compares the performance of the base Gemma-2b model with a fine-tuned version on CVPR papers data. The evaluation was conducted on 10 samples from the dataset.

## Summary of Results

### Perplexity (lower is better)
- **Base Model:** 14.11
- **Fine-tuned Model:** 10.10
- **Improvement:** 28.39%

### BLEU Scores (higher is better)
| Metric | Base Model | Fine-tuned Model | Improvement |
|--------|------------|------------------|-------------|
| BLEU-1 | 0.3350 | 0.3501 | 4.50% |
| BLEU-2 | 0.0585 | 0.0794 | 35.62% |
| BLEU-3 | 0.0158 | 0.0259 | 64.14% |
| BLEU-4 | 0.0086 | 0.0052 | -39.98% |

### ROUGE Scores (higher is better)
#### ROUGE-1
| Metric | Base Model | Fine-tuned Model | Improvement |
|--------|------------|------------------|-------------|
| F1 | 0.3427 | 0.3617 | 5.54% |
| Precision | 0.3793 | 0.3968 | 4.60% |
| Recall | 0.3141 | 0.3331 | 6.04% |

#### ROUGE-2
| Metric | Base Model | Fine-tuned Model | Improvement |
|--------|------------|------------------|-------------|
| F1 | 0.0739 | 0.0894 | 21.01% |
| Precision | 0.0820 | 0.0985 | 20.15% |
| Recall | 0.0677 | 0.0822 | 21.36% |

#### ROUGE-L
| Metric | Base Model | Fine-tuned Model | Improvement |
|--------|------------|------------------|-------------|
| F1 | 0.2058 | 0.2176 | 5.77% |
| Precision | 0.2274 | 0.2387 | 5.00% |
| Recall | 0.1889 | 0.2005 | 6.15% |

### Win Counts
| Metric | Base Model Wins | Fine-tuned Model Wins | Ties |
|--------|-----------------|----------------------|------|
| Perplexity | 0 | 10 | 0 |
| BLEU | 2 | 8 | 0 |
| ROUGE | 4 | 6 | 0 |

## Interpretation

### Perplexity
The fine-tuned model shows **lower perplexity**, indicating better prediction of CVPR paper text patterns.

### BLEU Scores
The fine-tuned model demonstrates **higher BLEU scores**, suggesting better n-gram precision when generating text that matches the reference.

### ROUGE Scores
The fine-tuned model achieves **higher ROUGE scores**, indicating better overlap with reference texts in terms of unigrams, bigrams, and longest common subsequences.

## Conclusion
Overall, the fine-tuning process has successfully improved the model's ability to generate text that matches the style and content patterns of CVPR papers. The improvements in perplexity, BLEU, and ROUGE scores demonstrate that the model has learned domain-specific knowledge and language patterns.

## Appendix
All metric calculations were performed using standard implementations:
- **Perplexity**: Calculated as exp(loss) on the masked language modeling task
- **BLEU**: Computed using a simplified implementation that calculates n-gram precision
- **ROUGE**: Calculated using Google's rouge-score package, including unigram overlap (ROUGE-1), bigram overlap (ROUGE-2), and longest common subsequence (ROUGE-L)

---
*Note: The plots corresponding to these metrics can be found in the evaluation_plots directory.*
