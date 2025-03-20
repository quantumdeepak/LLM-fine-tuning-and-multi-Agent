import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cvpr_comparison.log"),
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
    input_length = len(tokenizer.encode(prompt, add_special_tokens=False))
    generated_text = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
    
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

def calculate_text_similarity(text1, text2):
    """Calculate simple word overlap between two texts"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    overlap = len(words1.intersection(words2))
    total = len(words1.union(words2))
    
    return overlap / total if total > 0 else 0.0

def evaluate_models_on_cvpr_data(base_model, finetuned_model, tokenizer, evaluation_samples):
    """Compare model performance on CVPR papers data"""
    results = {
        "samples": [],
        "metrics": {
            "perplexity": {
                "base_model": [],
                "finetuned_model": []
            },
            "target_similarity": {
                "base_model": [],
                "finetuned_model": []
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
        
        # Calculate similarity between generated text and target
        base_similarity = calculate_text_similarity(base_continuation, target)
        finetuned_similarity = calculate_text_similarity(finetuned_continuation, target)
        
        # Log sample info
        if i < 3:  # Log only first 3 examples to avoid too much output
            logger.info(f"\nSample {i+1}: {title}")
            logger.info(f"Prompt: {prompt[:100]}...")
            logger.info(f"Target: {target[:100]}...")
            logger.info(f"Base model: {base_continuation[:100]}...")
            logger.info(f"Fine-tuned model: {finetuned_continuation[:100]}...")
            logger.info(f"Base perplexity: {base_perplexity:.2f}, Fine-tuned perplexity: {finetuned_perplexity:.2f}")
            logger.info(f"Base similarity: {base_similarity:.4f}, Fine-tuned similarity: {finetuned_similarity:.4f}")
        
        # Store results
        results["samples"].append({
            "title": title,
            "prompt": prompt,
            "target": target,
            "base_model_continuation": base_continuation,
            "finetuned_model_continuation": finetuned_continuation,
            "base_perplexity": base_perplexity,
            "finetuned_perplexity": finetuned_perplexity,
            "base_similarity": base_similarity,
            "finetuned_similarity": finetuned_similarity
        })
        
        results["metrics"]["perplexity"]["base_model"].append(base_perplexity)
        results["metrics"]["perplexity"]["finetuned_model"].append(finetuned_perplexity)
        results["metrics"]["target_similarity"]["base_model"].append(base_similarity)
        results["metrics"]["target_similarity"]["finetuned_model"].append(finetuned_similarity)
    
    return results

def summarize_results(results):
    """Summarize the evaluation results"""
    # Calculate averages
    avg_base_perplexity = np.mean(results["metrics"]["perplexity"]["base_model"])
    avg_finetuned_perplexity = np.mean(results["metrics"]["perplexity"]["finetuned_model"])
    
    avg_base_similarity = np.mean(results["metrics"]["target_similarity"]["base_model"])
    avg_finetuned_similarity = np.mean(results["metrics"]["target_similarity"]["finetuned_model"])
    
    # Calculate improvements
    perplexity_improvement = ((avg_base_perplexity - avg_finetuned_perplexity) / avg_base_perplexity) * 100
    similarity_improvement = ((avg_finetuned_similarity - avg_base_similarity) / avg_base_similarity) * 100 if avg_base_similarity > 0 else float('inf')
    
    # Create summary
    summary = {
        "perplexity": {
            "base_model": float(avg_base_perplexity),
            "finetuned_model": float(avg_finetuned_perplexity),
            "improvement_percent": float(perplexity_improvement)
        },
        "target_similarity": {
            "base_model": float(avg_base_similarity),
            "finetuned_model": float(avg_finetuned_similarity),
            "improvement_percent": float(similarity_improvement)
        },
        "win_counts": {
            "perplexity": {
                "base_wins": sum(b < f for b, f in zip(
                    results["metrics"]["perplexity"]["base_model"],
                    results["metrics"]["perplexity"]["finetuned_model"]
                )),
                "finetuned_wins": sum(f < b for b, f in zip(
                    results["metrics"]["perplexity"]["base_model"],
                    results["metrics"]["perplexity"]["finetuned_model"]
                ))
            },
            "similarity": {
                "base_wins": sum(b > f for b, f in zip(
                    results["metrics"]["target_similarity"]["base_model"],
                    results["metrics"]["target_similarity"]["finetuned_model"]
                )),
                "finetuned_wins": sum(f > b for b, f in zip(
                    results["metrics"]["target_similarity"]["base_model"],
                    results["metrics"]["target_similarity"]["finetuned_model"]
                ))
            }
        }
    }
    
    return summary

def run_cvpr_comparison(json_file, base_model_id, finetuned_model_path, hf_token=None, num_samples=10):
    """Run the full comparison pipeline on CVPR papers data"""
    # Load and process CVPR data
    evaluation_samples = load_and_process_cvpr_data(json_file, max_samples=num_samples)
    
    if not evaluation_samples:
        logger.error("No evaluation samples could be created from the data. Aborting comparison.")
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
    
    # Log summary
    logger.info("\n=== Model Comparison Summary ===")
    logger.info(f"Base Model: {base_model_id}")
    logger.info(f"Fine-tuned Model: {finetuned_model_path}")
    logger.info(f"Number of evaluation samples: {len(evaluation_samples)}")
    
    logger.info("\nPerplexity (lower is better):")
    logger.info(f"  Base Model: {summary['perplexity']['base_model']:.2f}")
    logger.info(f"  Fine-tuned Model: {summary['perplexity']['finetuned_model']:.2f}")
    
    if summary['perplexity']['improvement_percent'] > 0:
        logger.info(f"  Improvement: {summary['perplexity']['improvement_percent']:.2f}%")
    else:
        logger.info(f"  Degradation: {-summary['perplexity']['improvement_percent']:.2f}%")
    
    logger.info(f"  Win count - Base: {summary['win_counts']['perplexity']['base_wins']}, Fine-tuned: {summary['win_counts']['perplexity']['finetuned_wins']}")
    
    logger.info("\nTarget Text Similarity (higher is better):")
    logger.info(f"  Base Model: {summary['target_similarity']['base_model']:.4f}")
    logger.info(f"  Fine-tuned Model: {summary['target_similarity']['finetuned_model']:.4f}")
    
    if summary['target_similarity']['improvement_percent'] > 0:
        logger.info(f"  Improvement: {summary['target_similarity']['improvement_percent']:.2f}%")
    else:
        logger.info(f"  Degradation: {-summary['target_similarity']['improvement_percent']:.2f}%")
    
    logger.info(f"  Win count - Base: {summary['win_counts']['similarity']['base_wins']}, Fine-tuned: {summary['win_counts']['similarity']['finetuned_wins']}")
    
    # Save full results to file
    with open("cvpr_model_comparison_results.json", "w") as f:
        json.dump({
            "summary": summary,
            "samples": results["samples"]
        }, f, indent=2)
    
    logger.info("\nDetailed results saved to cvpr_model_comparison_results.json")
    
    return summary, results

if __name__ == "__main__":
    # Configuration
    INPUT_JSON = "/mnt/DATA/Glucoma/LLM/Testing_vllm/cvpr_papers.json"
    BASE_MODEL_ID = "google/gemma-2b"
    FINETUNED_MODEL_PATH = "fine_tuned_model/gemma_lora_finetuned"
    HF_TOKEN = os.getenv("HF_TOKEN")
    NUM_EVAL_SAMPLES = 10  # Number of papers to evaluate
    
    # Run comparison
    summary, results = run_cvpr_comparison(
        json_file=INPUT_JSON,
        base_model_id=BASE_MODEL_ID,
        finetuned_model_path=FINETUNED_MODEL_PATH,
        hf_token=HF_TOKEN,
        num_samples=NUM_EVAL_SAMPLES
    )