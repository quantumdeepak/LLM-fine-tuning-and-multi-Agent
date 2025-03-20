import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging
import random
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("comprehensive_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_and_process_cvpr_data(json_file, max_samples=10):
    """Load and process CVPR papers data for evaluation"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Successfully loaded JSON file: {json_file}")
        
        if not isinstance(data, list):
            logger.error(f"Expected JSON data to be a list, got {type(data)}")
            return []
            
        logger.info(f"Number of papers in dataset: {len(data)}")
        
        evaluation_samples = []
        
        # Process each paper
        for paper in data:
            if not isinstance(paper, dict) or 'text' not in paper:
                continue
                
            text = paper.get('text', '')
            if not isinstance(text, str) or len(text) < 100:  # Skip very short texts
                continue
                
            # Extract title if available
            title = paper.get('title', '')
            
            # Split text into chunks of appropriate size
            # We'll use the start of each paper as a prompt and test continuation
            words = text.split()
            if len(words) < 50:  # Skip very short papers
                continue
                
            # Take first 30-50 words as prompt
            prompt_length = min(50, len(words) // 4)  # Take ~25% of text as prompt, max 50 words
            prompt = ' '.join(words[:prompt_length])
            
            # Take next 100 words as reference text for comparison
            target_length = min(100, len(words) - prompt_length)
            target = ' '.join(words[prompt_length:prompt_length + target_length])
            
            evaluation_samples.append({
                'title': title,
                'prompt': prompt,
                'target': target,
                'full_text': text
            })
        
        # Select a random subset if we have more than max_samples
        if len(evaluation_samples) > max_samples:
            evaluation_samples = random.sample(evaluation_samples, max_samples)
            
        logger.info(f"Created {len(evaluation_samples)} evaluation samples")
        
        return evaluation_samples
    except Exception as e:
        logger.error(f"Error processing JSON file: {e}")
        return []

def load_models_for_comparison(base_model_id, finetuned_model_path, hf_token=None):
    """Load both the base pre-trained model and the fine-tuned model"""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        token=hf_token,
        trust_remote_code=True
    )
    
    # Load base model
    logger.info(f"Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        token=hf_token,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # Load fine-tuned model
    logger.info(f"Loading fine-tuned model from: {finetuned_model_path}")
    # Load the base model first
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        token=hf_token,
        trust_remote_code=True,
        device_map="auto"
    )
    # Then load the LoRA weights
    finetuned_model = PeftModel.from_pretrained(
        finetuned_model,
        finetuned_model_path
    )
    
    return tokenizer, base_model, finetuned_model

def generate_text(model, tokenizer, prompt, max_length=100):
    """Generate text from a prompt"""
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate text
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    # Get the full generated text
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract just the newly generated portion
    prompt_length = len(tokenizer.encode(prompt, add_special_tokens=True)) - 2  # Adjust for special tokens
    generated_text = tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True)
    
    return generated_text.strip()

def calculate_perplexity(model, tokenizer, text, max_length=512):
    """Calculate perplexity of a model on given text"""
    # Tokenize text, truncating if necessary
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    
    # Move to device
    input_ids = encodings.input_ids.to(model.device)
    
    # Calculate perplexity
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        neg_log_likelihood = outputs.loss
    
    # Perplexity = exp(negative log likelihood)
    perplexity = torch.exp(neg_log_likelihood).item()
    
    return perplexity

def calculate_simple_bleu(reference, candidate):
    """
    Calculate a simplified BLEU score without requiring NLTK
    This uses a custom implementation for n-gram precision
    """
    # Simple tokenization by splitting on whitespace
    reference_tokens = reference.lower().split()
    candidate_tokens = candidate.lower().split()
    
    if not candidate_tokens:
        return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}
    
    # Calculate n-gram precision for different n values
    bleu_scores = {}
    
    for n in range(1, 5):
        if len(candidate_tokens) < n:
            bleu_scores[f"bleu_{n}"] = 0.0
            continue
            
        # Create n-grams
        ref_ngrams = {}
        for i in range(len(reference_tokens) - n + 1):
            ngram = tuple(reference_tokens[i:i+n])
            ref_ngrams[ngram] = ref_ngrams.get(ngram, 0) + 1
            
        cand_ngrams = {}
        for i in range(len(candidate_tokens) - n + 1):
            ngram = tuple(candidate_tokens[i:i+n])
            cand_ngrams[ngram] = cand_ngrams.get(ngram, 0) + 1
        
        # Count matches (clipped by reference counts)
        matches = 0
        for ngram, count in cand_ngrams.items():
            matches += min(count, ref_ngrams.get(ngram, 0))
        
        # Calculate precision
        total_ngrams = max(1, len(candidate_tokens) - n + 1)  # avoid division by zero
        precision = matches / total_ngrams
        
        bleu_scores[f"bleu_{n}"] = precision
    
    return bleu_scores

def calculate_rouge_scores(reference, candidate):
    """Calculate ROUGE scores between reference and candidate texts"""
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        
        return {
            "rouge_1": {
                "precision": scores['rouge1'].precision,
                "recall": scores['rouge1'].recall,
                "f1": scores['rouge1'].fmeasure
            },
            "rouge_2": {
                "precision": scores['rouge2'].precision,
                "recall": scores['rouge2'].recall,
                "f1": scores['rouge2'].fmeasure
            },
            "rouge_l": {
                "precision": scores['rougeL'].precision,
                "recall": scores['rougeL'].recall,
                "f1": scores['rougeL'].fmeasure
            }
        }
    except Exception as e:
        logger.warning(f"Error calculating ROUGE scores: {e}")
        # Return default values if ROUGE calculation fails
        return {
            "rouge_1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "rouge_2": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "rouge_l": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        }

def evaluate_models_on_cvpr_data(base_model, finetuned_model, tokenizer, evaluation_samples):
    """Compare model performance on CVPR papers data"""
    results = {
        "samples": [],
        "metrics": {
            "perplexity": {
                "base_model": [],
                "finetuned_model": []
            },
            "bleu": {
                "base_model": {
                    "bleu_1": [], "bleu_2": [], "bleu_3": [], "bleu_4": []
                },
                "finetuned_model": {
                    "bleu_1": [], "bleu_2": [], "bleu_3": [], "bleu_4": []
                }
            },
            "rouge": {
                "base_model": {
                    "rouge_1": {"f1": [], "precision": [], "recall": []},
                    "rouge_2": {"f1": [], "precision": [], "recall": []},
                    "rouge_l": {"f1": [], "precision": [], "recall": []}
                },
                "finetuned_model": {
                    "rouge_1": {"f1": [], "precision": [], "recall": []},
                    "rouge_2": {"f1": [], "precision": [], "recall": []},
                    "rouge_l": {"f1": [], "precision": [], "recall": []}
                }
            }
        }
    }
    
    logger.info("Evaluating models on CVPR papers...")
    
    for i, sample in enumerate(tqdm(evaluation_samples)):
        prompt = sample["prompt"]
        target = sample["target"]
        title = sample.get("title", f"Paper {i+1}")
        
        # Generate continuations from both models
        base_continuation = generate_text(base_model, tokenizer, prompt)
        finetuned_continuation = generate_text(finetuned_model, tokenizer, prompt)
        
        # Calculate perplexity on prompt + target
        prompt_target = prompt + " " + target
        base_perplexity = calculate_perplexity(base_model, tokenizer, prompt_target)
        finetuned_perplexity = calculate_perplexity(finetuned_model, tokenizer, prompt_target)
        
        # Calculate BLEU scores
        base_bleu = calculate_simple_bleu(target, base_continuation)
        finetuned_bleu = calculate_simple_bleu(target, finetuned_continuation)
        
        # Calculate ROUGE scores
        base_rouge = calculate_rouge_scores(target, base_continuation)
        finetuned_rouge = calculate_rouge_scores(target, finetuned_continuation)
        
        # Log sample info
        if i < 3:  # Log only first 3 examples to avoid too much output
            logger.info(f"\nSample {i+1}: {title}")
            logger.info(f"Prompt: {prompt[:100]}...")
            logger.info(f"Target: {target[:100]}...")
            logger.info(f"Base model: {base_continuation[:100]}...")
            logger.info(f"Fine-tuned model: {finetuned_continuation[:100]}...")
            logger.info(f"Base perplexity: {base_perplexity:.2f}, Fine-tuned perplexity: {finetuned_perplexity:.2f}")
            logger.info(f"Base BLEU-1: {base_bleu['bleu_1']:.4f}, Fine-tuned BLEU-1: {finetuned_bleu['bleu_1']:.4f}")
            logger.info(f"Base ROUGE-1 F1: {base_rouge['rouge_1']['f1']:.4f}, Fine-tuned ROUGE-1 F1: {finetuned_rouge['rouge_1']['f1']:.4f}")
        
        # Store results
        sample_result = {
            "title": title,
            "prompt": prompt,
            "target": target,
            "base_model_continuation": base_continuation,
            "finetuned_model_continuation": finetuned_continuation,
            "base_perplexity": base_perplexity,
            "finetuned_perplexity": finetuned_perplexity,
            "base_bleu": base_bleu,
            "finetuned_bleu": finetuned_bleu,
            "base_rouge": base_rouge,
            "finetuned_rouge": finetuned_rouge
        }
        
        results["samples"].append(sample_result)
        
        # Store metrics
        results["metrics"]["perplexity"]["base_model"].append(base_perplexity)
        results["metrics"]["perplexity"]["finetuned_model"].append(finetuned_perplexity)
        
        # Store BLEU scores
        for k in ["bleu_1", "bleu_2", "bleu_3", "bleu_4"]:
            results["metrics"]["bleu"]["base_model"][k].append(base_bleu[k])
            results["metrics"]["bleu"]["finetuned_model"][k].append(finetuned_bleu[k])
        
        # Store ROUGE scores
        for rouge_type in ["rouge_1", "rouge_2", "rouge_l"]:
            for metric in ["f1", "precision", "recall"]:
                results["metrics"]["rouge"]["base_model"][rouge_type][metric].append(base_rouge[rouge_type][metric])
                results["metrics"]["rouge"]["finetuned_model"][rouge_type][metric].append(finetuned_rouge[rouge_type][metric])
    
    return results

def summarize_results(results):
    """Summarize the evaluation results"""
    summary = {
        "perplexity": {
            "base_model": float(np.mean(results["metrics"]["perplexity"]["base_model"])),
            "finetuned_model": float(np.mean(results["metrics"]["perplexity"]["finetuned_model"]))
        },
        "bleu": {
            "base_model": {},
            "finetuned_model": {}
        },
        "rouge": {
            "base_model": {},
            "finetuned_model": {}
        },
        "win_counts": {
            "perplexity": {
                "base_wins": 0,
                "finetuned_wins": 0,
                "ties": 0
            },
            "bleu": {
                "base_wins": 0,
                "finetuned_wins": 0,
                "ties": 0
            },
            "rouge": {
                "base_wins": 0,
                "finetuned_wins": 0,
                "ties": 0
            }
        }
    }
    
    # Calculate average BLEU scores
    for k in ["bleu_1", "bleu_2", "bleu_3", "bleu_4"]:
        summary["bleu"]["base_model"][k] = float(np.mean(results["metrics"]["bleu"]["base_model"][k]))
        summary["bleu"]["finetuned_model"][k] = float(np.mean(results["metrics"]["bleu"]["finetuned_model"][k]))
    
    # Calculate average ROUGE scores
    for rouge_type in ["rouge_1", "rouge_2", "rouge_l"]:
        if rouge_type not in summary["rouge"]["base_model"]:
            summary["rouge"]["base_model"][rouge_type] = {}
            summary["rouge"]["finetuned_model"][rouge_type] = {}
            
        for metric in ["f1", "precision", "recall"]:
            summary["rouge"]["base_model"][rouge_type][metric] = float(np.mean(results["metrics"]["rouge"]["base_model"][rouge_type][metric]))
            summary["rouge"]["finetuned_model"][rouge_type][metric] = float(np.mean(results["metrics"]["rouge"]["finetuned_model"][rouge_type][metric]))
    
    # Calculate win counts for perplexity (lower is better)
    for base_perp, finetuned_perp in zip(
        results["metrics"]["perplexity"]["base_model"],
        results["metrics"]["perplexity"]["finetuned_model"]
    ):
        if base_perp < finetuned_perp:
            summary["win_counts"]["perplexity"]["base_wins"] += 1
        elif finetuned_perp < base_perp:
            summary["win_counts"]["perplexity"]["finetuned_wins"] += 1
        else:
            summary["win_counts"]["perplexity"]["ties"] += 1
    
    # Calculate win counts for BLEU-1 (higher is better)
    for base_bleu, finetuned_bleu in zip(
        results["metrics"]["bleu"]["base_model"]["bleu_1"],
        results["metrics"]["bleu"]["finetuned_model"]["bleu_1"]
    ):
        if base_bleu > finetuned_bleu:
            summary["win_counts"]["bleu"]["base_wins"] += 1
        elif finetuned_bleu > base_bleu:
            summary["win_counts"]["bleu"]["finetuned_wins"] += 1
        else:
            summary["win_counts"]["bleu"]["ties"] += 1
    
    # Calculate win counts for ROUGE-1 F1 (higher is better)
    for base_rouge, finetuned_rouge in zip(
        results["metrics"]["rouge"]["base_model"]["rouge_1"]["f1"],
        results["metrics"]["rouge"]["finetuned_model"]["rouge_1"]["f1"]
    ):
        if base_rouge > finetuned_rouge:
            summary["win_counts"]["rouge"]["base_wins"] += 1
        elif finetuned_rouge > base_rouge:
            summary["win_counts"]["rouge"]["finetuned_wins"] += 1
        else:
            summary["win_counts"]["rouge"]["ties"] += 1
    
    # Calculate improvement percentages
    perplexity_improvement = ((summary["perplexity"]["base_model"] - summary["perplexity"]["finetuned_model"]) / 
                             summary["perplexity"]["base_model"]) * 100
    
    bleu_improvements = {}
    for k in ["bleu_1", "bleu_2", "bleu_3", "bleu_4"]:
        if summary["bleu"]["base_model"][k] > 0:
            bleu_improvements[k] = ((summary["bleu"]["finetuned_model"][k] - summary["bleu"]["base_model"][k]) / 
                                   summary["bleu"]["base_model"][k]) * 100
        else:
            bleu_improvements[k] = float('inf') if summary["bleu"]["finetuned_model"][k] > 0 else 0
    
    rouge_improvements = {}
    for rouge_type in ["rouge_1", "rouge_2", "rouge_l"]:
        rouge_improvements[rouge_type] = {}
        for metric in ["f1", "precision", "recall"]:
            if summary["rouge"]["base_model"][rouge_type][metric] > 0:
                rouge_improvements[rouge_type][metric] = ((summary["rouge"]["finetuned_model"][rouge_type][metric] - 
                                                         summary["rouge"]["base_model"][rouge_type][metric]) / 
                                                        summary["rouge"]["base_model"][rouge_type][metric]) * 100
            else:
                rouge_improvements[rouge_type][metric] = float('inf') if summary["rouge"]["finetuned_model"][rouge_type][metric] > 0 else 0
    
    summary["improvements"] = {
        "perplexity": float(perplexity_improvement),
        "bleu": {k: float(v) for k, v in bleu_improvements.items()},
        "rouge": {rt: {m: float(v) for m, v in metrics.items()} for rt, metrics in rouge_improvements.items()}
    }
    
    return summary

def generate_plots(summary, output_dir="evaluation_plots"):
    """Generate plots to visualize the comparison results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the style
    plt.style.use('ggplot')
    
    # 1. Perplexity comparison
    plt.figure(figsize=(10, 6))
    models = ['Base Model', 'Fine-tuned Model']
    perplexities = [summary["perplexity"]["base_model"], summary["perplexity"]["finetuned_model"]]
    
    bars = plt.bar(models, perplexities, color=['#3498db', '#2ecc71'])
    plt.ylabel('Perplexity (lower is better)')
    plt.title('Perplexity Comparison')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/perplexity_comparison.png", dpi=300)
    plt.close()
    
    # 2. BLEU score comparison
    plt.figure(figsize=(12, 7))
    
    bleu_metrics = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]
    base_bleu = [summary["bleu"]["base_model"][f"bleu_{i}"] for i in range(1, 5)]
    finetuned_bleu = [summary["bleu"]["finetuned_model"][f"bleu_{i}"] for i in range(1, 5)]
    
    x = np.arange(len(bleu_metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, base_bleu, width, label='Base Model', color='#3498db')
    rects2 = ax.bar(x + width/2, finetuned_bleu, width, label='Fine-tuned Model', color='#2ecc71')
    
    ax.set_ylabel('Score (higher is better)')
    ax.set_title('BLEU Score Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(bleu_metrics)
    ax.legend()
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig(f"{output_dir}/bleu_comparison.png", dpi=300)
    plt.close()
    
    # 3. ROUGE F1 scores comparison
    plt.figure(figsize=(12, 7))
    
    rouge_metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    base_rouge = [summary["rouge"]["base_model"][f"rouge_{x.lower().replace('-', '')}"]["f1"] for x in ["1", "2", "l"]]
    finetuned_rouge = [summary["rouge"]["finetuned_model"][f"rouge_{x.lower().replace('-', '')}"]["f1"] for x in ["1", "2", "l"]]
    
    x = np.arange(len(rouge_metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, base_rouge, width, label='Base Model', color='#3498db')
    rects2 = ax.bar(x + width/2, finetuned_rouge, width, label='Fine-tuned Model', color='#2ecc71')
    
    ax.set_ylabel('F1 Score (higher is better)')
    ax.set_title('ROUGE F1 Score Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(rouge_metrics)
    ax.legend()
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig(f"{output_dir}/rouge_f1_comparison.png", dpi=300)
    plt.close()
    
    # 4. Win counts
    plt.figure(figsize=(12, 7))
    
    metrics = ["Perplexity", "BLEU", "ROUGE"]
    base_wins = [summary["win_counts"][m.lower()]["base_wins"] for m in metrics]
    finetuned_wins = [summary["win_counts"][m.lower()]["finetuned_wins"] for m in metrics]
    ties = [summary["win_counts"][m.lower()]["ties"] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width, base_wins, width, label='Base Model Wins', color='#3498db')
    rects2 = ax.bar(x, finetuned_wins, width, label='Fine-tuned Model Wins', color='#2ecc71')
    rects3 = ax.bar(x + width, ties, width, label='Ties', color='#95a5a6')
    
    ax.set_ylabel('Number of Samples')
    ax.set_title('Win Count Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    fig.tight_layout()
    plt.savefig(f"{output_dir}/win_counts.png", dpi=300)
    plt.close()
    
    logger.info(f"Plots saved to {output_dir} directory")

def generate_markdown_report(summary, num_samples, output_file="model_comparison_report.md"):
    """Generate a comprehensive markdown report of the evaluation results"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# Model Comparison Report
*Generated on: {timestamp}*

## Overview
This report compares the performance of the base Gemma-2b model with a fine-tuned version on CVPR papers data. The evaluation was conducted on {num_samples} samples from the dataset.

## Summary of Results

### Perplexity (lower is better)
- **Base Model:** {summary["perplexity"]["base_model"]:.2f}
- **Fine-tuned Model:** {summary["perplexity"]["finetuned_model"]:.2f}
- **Improvement:** {summary["improvements"]["perplexity"]:.2f}%

### BLEU Scores (higher is better)
| Metric | Base Model | Fine-tuned Model | Improvement |
|--------|------------|------------------|-------------|
| BLEU-1 | {summary["bleu"]["base_model"]["bleu_1"]:.4f} | {summary["bleu"]["finetuned_model"]["bleu_1"]:.4f} | {summary["improvements"]["bleu"]["bleu_1"]:.2f}% |
| BLEU-2 | {summary["bleu"]["base_model"]["bleu_2"]:.4f} | {summary["bleu"]["finetuned_model"]["bleu_2"]:.4f} | {summary["improvements"]["bleu"]["bleu_2"]:.2f}% |
| BLEU-3 | {summary["bleu"]["base_model"]["bleu_3"]:.4f} | {summary["bleu"]["finetuned_model"]["bleu_3"]:.4f} | {summary["improvements"]["bleu"]["bleu_3"]:.2f}% |
| BLEU-4 | {summary["bleu"]["base_model"]["bleu_4"]:.4f} | {summary["bleu"]["finetuned_model"]["bleu_4"]:.4f} | {summary["improvements"]["bleu"]["bleu_4"]:.2f}% |

### ROUGE Scores (higher is better)
#### ROUGE-1
| Metric | Base Model | Fine-tuned Model | Improvement |
|--------|------------|------------------|-------------|
| F1 | {summary["rouge"]["base_model"]["rouge_1"]["f1"]:.4f} | {summary["rouge"]["finetuned_model"]["rouge_1"]["f1"]:.4f} | {summary["improvements"]["rouge"]["rouge_1"]["f1"]:.2f}% |
| Precision | {summary["rouge"]["base_model"]["rouge_1"]["precision"]:.4f} | {summary["rouge"]["finetuned_model"]["rouge_1"]["precision"]:.4f} | {summary["improvements"]["rouge"]["rouge_1"]["precision"]:.2f}% |
| Recall | {summary["rouge"]["base_model"]["rouge_1"]["recall"]:.4f} | {summary["rouge"]["finetuned_model"]["rouge_1"]["recall"]:.4f} | {summary["improvements"]["rouge"]["rouge_1"]["recall"]:.2f}% |

#### ROUGE-2
| Metric | Base Model | Fine-tuned Model | Improvement |
|--------|------------|------------------|-------------|
| F1 | {summary["rouge"]["base_model"]["rouge_2"]["f1"]:.4f} | {summary["rouge"]["finetuned_model"]["rouge_2"]["f1"]:.4f} | {summary["improvements"]["rouge"]["rouge_2"]["f1"]:.2f}% |
| Precision | {summary["rouge"]["base_model"]["rouge_2"]["precision"]:.4f} | {summary["rouge"]["finetuned_model"]["rouge_2"]["precision"]:.4f} | {summary["improvements"]["rouge"]["rouge_2"]["precision"]:.2f}% |
| Recall | {summary["rouge"]["base_model"]["rouge_2"]["recall"]:.4f} | {summary["rouge"]["finetuned_model"]["rouge_2"]["recall"]:.4f} | {summary["improvements"]["rouge"]["rouge_2"]["recall"]:.2f}% |

#### ROUGE-L
| Metric | Base Model | Fine-tuned Model | Improvement |
|--------|------------|------------------|-------------|
| F1 | {summary["rouge"]["base_model"]["rouge_l"]["f1"]:.4f} | {summary["rouge"]["finetuned_model"]["rouge_l"]["f1"]:.4f} | {summary["improvements"]["rouge"]["rouge_l"]["f1"]:.2f}% |
| Precision | {summary["rouge"]["base_model"]["rouge_l"]["precision"]:.4f} | {summary["rouge"]["finetuned_model"]["rouge_l"]["precision"]:.4f} | {summary["improvements"]["rouge"]["rouge_l"]["precision"]:.2f}% |
| Recall | {summary["rouge"]["base_model"]["rouge_l"]["recall"]:.4f} | {summary["rouge"]["finetuned_model"]["rouge_l"]["recall"]:.4f} | {summary["improvements"]["rouge"]["rouge_l"]["recall"]:.2f}% |

### Win Counts
| Metric | Base Model Wins | Fine-tuned Model Wins | Ties |
|--------|-----------------|----------------------|------|
| Perplexity | {summary["win_counts"]["perplexity"]["base_wins"]} | {summary["win_counts"]["perplexity"]["finetuned_wins"]} | {summary["win_counts"]["perplexity"]["ties"]} |
| BLEU | {summary["win_counts"]["bleu"]["base_wins"]} | {summary["win_counts"]["bleu"]["finetuned_wins"]} | {summary["win_counts"]["bleu"]["ties"]} |
| ROUGE | {summary["win_counts"]["rouge"]["base_wins"]} | {summary["win_counts"]["rouge"]["finetuned_wins"]} | {summary["win_counts"]["rouge"]["ties"]} |

## Interpretation

### Perplexity
{'The fine-tuned model shows **lower perplexity**, indicating better prediction of CVPR paper text patterns.' if summary["perplexity"]["finetuned_model"] < summary["perplexity"]["base_model"] else 'The base model shows lower perplexity on this dataset.'}

### BLEU Scores
{'The fine-tuned model demonstrates **higher BLEU scores**, suggesting better n-gram precision when generating text that matches the reference.' if summary["bleu"]["finetuned_model"]["bleu_1"] > summary["bleu"]["base_model"]["bleu_1"] else 'The base model achieves higher BLEU scores on this dataset.'}

### ROUGE Scores
{'The fine-tuned model achieves **higher ROUGE scores**, indicating better overlap with reference texts in terms of unigrams, bigrams, and longest common subsequences.' if summary["rouge"]["finetuned_model"]["rouge_1"]["f1"] > summary["rouge"]["base_model"]["rouge_1"]["f1"] else 'The base model achieves higher ROUGE scores on this dataset.'}

## Conclusion
{'Overall, the fine-tuning process has successfully improved the model\'s ability to generate text that matches the style and content patterns of CVPR papers. The improvements in perplexity, BLEU, and ROUGE scores demonstrate that the model has learned domain-specific knowledge and language patterns.' if (summary["perplexity"]["finetuned_model"] < summary["perplexity"]["base_model"] and summary["bleu"]["finetuned_model"]["bleu_1"] > summary["bleu"]["base_model"]["bleu_1"]) else 'The results show mixed performance between the base and fine-tuned models. Further fine-tuning with optimized hyperparameters or additional data might be needed to achieve consistent improvements across all metrics.'}

## Appendix
All metric calculations were performed using standard implementations:
- **Perplexity**: Calculated as exp(loss) on the masked language modeling task
- **BLEU**: Computed using a simplified implementation that calculates n-gram precision
- **ROUGE**: Calculated using Google's rouge-score package, including unigram overlap (ROUGE-1), bigram overlap (ROUGE-2), and longest common subsequence (ROUGE-L)

---
*Note: The plots corresponding to these metrics can be found in the evaluation_plots directory.*
"""
    
    # Write report to file
    with open(output_file, 'w') as f:
        f.write(report)
    
    logger.info(f"Markdown report saved to {output_file}")
    
    return report

def run_comprehensive_evaluation(json_file, base_model_id, finetuned_model_path, hf_token=None, num_samples=10):
    """Run a comprehensive evaluation pipeline on CVPR papers data"""
    # Load and process CVPR data
    evaluation_samples = load_and_process_cvpr_data(json_file, max_samples=num_samples)
    
    if not evaluation_samples:
        logger.error("No evaluation samples could be created from the data. Aborting evaluation.")
        return None
    
    # Load models
    tokenizer, base_model, finetuned_model = load_models_for_comparison(
        base_model_id, 
        finetuned_model_path, 
        hf_token
    )
    
    # Evaluate models
    results = evaluate_models_on_cvpr_data(base_model, finetuned_model, tokenizer, evaluation_samples)
    
    # Summarize results
    summary = summarize_results(results)
    
    # Generate plots
    generate_plots(summary)
    
    # Generate markdown report
    generate_markdown_report(summary, len(evaluation_samples))
    
    # Log summary
    logger.info("\n=== Model Comparison Summary ===")
    logger.info(f"Base Model: {base_model_id}")
    logger.info(f"Fine-tuned Model: {finetuned_model_path}")
    logger.info(f"Number of evaluation samples: {len(evaluation_samples)}")
    
    logger.info("\nPerplexity (lower is better):")
    logger.info(f"  Base Model: {summary['perplexity']['base_model']:.2f}")
    logger.info(f"  Fine-tuned Model: {summary['perplexity']['finetuned_model']:.2f}")
    
    if summary['improvements']['perplexity'] > 0:
        logger.info(f"  Improvement: {summary['improvements']['perplexity']:.2f}%")
    else:
        logger.info(f"  Degradation: {-summary['improvements']['perplexity']:.2f}%")
    
    logger.info("\nBLEU-1 Score (higher is better):")
    logger.info(f"  Base Model: {summary['bleu']['base_model']['bleu_1']:.4f}")
    logger.info(f"  Fine-tuned Model: {summary['bleu']['finetuned_model']['bleu_1']:.4f}")
    
    if summary['improvements']['bleu']['bleu_1'] > 0:
        logger.info(f"  Improvement: {summary['improvements']['bleu']['bleu_1']:.2f}%")
    else:
        logger.info(f"  Degradation: {-summary['improvements']['bleu']['bleu_1']:.2f}%")
    
    logger.info("\nROUGE-1 F1 (higher is better):")
    logger.info(f"  Base Model: {summary['rouge']['base_model']['rouge_1']['f1']:.4f}")
    logger.info(f"  Fine-tuned Model: {summary['rouge']['finetuned_model']['rouge_1']['f1']:.4f}")
    
    if summary['improvements']['rouge']['rouge_1']['f1'] > 0:
        logger.info(f"  Improvement: {summary['improvements']['rouge']['rouge_1']['f1']:.2f}%")
    else:
        logger.info(f"  Degradation: {-summary['improvements']['rouge']['rouge_1']['f1']:.2f}%")
    
    # Save full results to file
    with open("comprehensive_evaluation_results.json", "w") as f:
        json.dump({
            "summary": summary,
            "samples": results["samples"]
        }, f, indent=2)
    
    logger.info("\nDetailed results saved to comprehensive_evaluation_results.json")
    logger.info("Markdown report generated: model_comparison_report.md")
    logger.info("Plots generated in the evaluation_plots directory")
    
    return summary, results

if __name__ == "__main__":
    # Configuration
    INPUT_JSON = "/mnt/DATA/Glucoma/LLM/Testing_vllm/cvpr_papers.json"
    BASE_MODEL_ID = "google/gemma-2b"
    FINETUNED_MODEL_PATH = "fine_tuned_model/gemma_lora_finetuned"
    HF_TOKEN = os.getenv("HF_TOKEN")
    NUM_EVAL_SAMPLES = 10  # Number of papers to evaluate
    
    # Run comprehensive evaluation
    try:
        summary, results = run_comprehensive_evaluation(
            json_file=INPUT_JSON,
            base_model_id=BASE_MODEL_ID,
            finetuned_model_path=FINETUNED_MODEL_PATH,
            hf_token=HF_TOKEN,
            num_samples=NUM_EVAL_SAMPLES
        )
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())