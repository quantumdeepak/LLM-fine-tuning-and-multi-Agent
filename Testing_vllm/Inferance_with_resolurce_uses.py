import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import logging
import time
import psutil
import threading
import datetime
from pathlib import Path

# ############################################################
# ################ CONFIGURATION PARAMETERS ##################
# ############################################################

CONFIG = {
    # Model parameters
    "base_model": "google/gemma-2b",         # Base model ID from Hugging Face
    "lora_weights": "fine_tuned_model/gemma_lora_finetuned",  # Path to LoRA weights
    "hf_token": os.getenv("HF_TOKEN", None), # Optional HF token for gated models
    
    # Device parameters
    "gpu_id": 0,                  # Which GPU to use (0, 1, 2, etc.)
    "use_cpu": False,             # Set to True to force CPU usage
    
    # Generation parameters
    "max_length": 512,            # Maximum tokens to generate
    "temperature": 0.7,           # Temperature for sampling
    "top_p": 0.9,                 # Top-p sampling value
    
    # Resource monitoring parameters
    "monitor_interval": 1.0,      # Seconds between resource checks
    "log_dir": "/mnt/DATA/Glucoma/LLM/Testing_vllm/logs",      # Directory to store logs
    "print_resources": True,      # Print resource usage after each generation
}

# Create derived paths
CONFIG["inference_log"] = os.path.join(CONFIG["log_dir"], "inference.log")
CONFIG["resource_log"] = os.path.join(CONFIG["log_dir"], "resources.log")

# ############################################################

# Create logs directory
os.makedirs(CONFIG["log_dir"], exist_ok=True)
print(f"Created log directory: {os.path.abspath(CONFIG['log_dir'])}")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG["inference_log"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create a separate logger for resource metrics
resource_logger = logging.getLogger("resource_metrics")
resource_logger.setLevel(logging.INFO)
resource_logger.propagate = False
resource_handler = logging.FileHandler(CONFIG["resource_log"])
resource_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
resource_logger.addHandler(resource_handler)

# Global flag for resource monitoring
monitor_resources = True

# Store latest resource metrics for reporting
latest_metrics = {
    "cpu_percent": 0,
    "memory_used_gb": 0,
    "memory_percent": 0,
    "gpu_allocated_gb": {},
    "gpu_reserved_gb": {}
}

def get_gpu_memory_info():
    """Get GPU memory usage information if available"""
    try:
        if torch.cuda.is_available():
            gpu_memory = {}
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)    # GB
                gpu_memory[f"gpu{i}"] = {
                    "allocated_gb": allocated,
                    "reserved_gb": reserved,
                    "name": torch.cuda.get_device_name(i)
                }
            return gpu_memory
        return {}
    except Exception as e:
        logger.error(f"Error getting GPU memory info: {e}")
        return {}

def get_cpu_memory_info():
    """Get CPU memory usage information"""
    try:
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024 ** 3),
            "available_gb": memory.available / (1024 ** 3),
            "used_gb": memory.used / (1024 ** 3),
            "percent": memory.percent
        }
    except Exception as e:
        logger.error(f"Error getting CPU memory info: {e}")
        return {}

def get_cpu_usage():
    """Get CPU usage percentage"""
    try:
        return psutil.cpu_percent(interval=0.1)
    except Exception as e:
        logger.error(f"Error getting CPU usage: {e}")
        return 0.0

def resource_monitoring_thread():
    """Thread function to monitor system resources"""
    print(f"Resource monitoring started. Logging to: {os.path.abspath(CONFIG['resource_log'])}")
    resource_logger.info("Starting resource monitoring")
    resource_logger.info("=" * 80)
    
    # Create header for CSV format
    header = "timestamp,event,cpu_percent,cpu_memory_used_gb,cpu_memory_percent"
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            header += f",gpu{i}_allocated_gb,gpu{i}_reserved_gb"
    resource_logger.info(header)
    
    # Log initial system state
    log_resource_metrics("monitoring_started")
    
    while monitor_resources:
        log_resource_metrics("periodic")
        time.sleep(CONFIG["monitor_interval"])
    
    # Log final system state
    log_resource_metrics("monitoring_stopped")
    resource_logger.info("=" * 80)
    resource_logger.info("Resource monitoring stopped")

def log_resource_metrics(event="periodic"):
    """Log current resource metrics"""
    global latest_metrics
    
    try:
        cpu_percent = get_cpu_usage()
        cpu_memory = get_cpu_memory_info()
        gpu_memory = get_gpu_memory_info()
        
        # Update latest metrics
        latest_metrics["cpu_percent"] = cpu_percent
        latest_metrics["memory_used_gb"] = cpu_memory.get('used_gb', 0)
        latest_metrics["memory_percent"] = cpu_memory.get('percent', 0)
        
        # Base metrics string
        metrics = f"{datetime.datetime.now().isoformat()},{event},{cpu_percent:.1f},{cpu_memory.get('used_gb', 0):.2f},{cpu_memory.get('percent', 0):.1f}"
        
        # Add GPU metrics if available
        if gpu_memory:
            for i in range(torch.cuda.device_count()):
                gpu_key = f"gpu{i}"
                if gpu_key in gpu_memory:
                    metrics += f",{gpu_memory[gpu_key]['allocated_gb']:.2f},{gpu_memory[gpu_key]['reserved_gb']:.2f}"
                    latest_metrics["gpu_allocated_gb"][gpu_key] = gpu_memory[gpu_key]['allocated_gb']
                    latest_metrics["gpu_reserved_gb"][gpu_key] = gpu_memory[gpu_key]['reserved_gb']
        
        resource_logger.info(metrics)
    except Exception as e:
        logger.error(f"Error logging resource metrics: {e}")

def print_resource_summary():
    """Print a summary of current resource usage"""
    print("\n----- RESOURCE USAGE -----")
    print(f"CPU Usage: {latest_metrics['cpu_percent']:.1f}%")
    print(f"Memory Usage: {latest_metrics['memory_used_gb']:.2f} GB ({latest_metrics['memory_percent']:.1f}%)")
    
    if latest_metrics["gpu_allocated_gb"]:
        for gpu_key in latest_metrics["gpu_allocated_gb"]:
            print(f"{gpu_key.upper()} Memory: {latest_metrics['gpu_allocated_gb'][gpu_key]:.2f} GB allocated, "
                  f"{latest_metrics['gpu_reserved_gb'][gpu_key]:.2f} GB reserved")
    
    print("Resource logs saved to:", os.path.abspath(CONFIG["resource_log"]))
    print("--------------------------\n")

def load_model_with_lora():
    logger.info(f"Loading base model: {CONFIG['base_model']}")
    log_resource_metrics("load_model_start")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model"],
        trust_remote_code=True,
        token=CONFIG["hf_token"]
    )
    
    # Determine device
    if CONFIG["use_cpu"]:
        device = torch.device("cpu")
        logger.info("Forcing CPU usage as specified in config")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{CONFIG['gpu_id']}")
            logger.info(f"Using GPU: {CONFIG['gpu_id']} ({torch.cuda.get_device_name(CONFIG['gpu_id'])})")
        else:
            device = torch.device("cpu")
            logger.info("No GPU available, using CPU")
    
    # Log current system resources
    log_resource_metrics("tokenizer_loaded")
    
    # Load base model
    logger.info("Loading base model (this may take a while)...")
    
    try:
        # First try with bfloat16 precision
        load_start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["base_model"],
            torch_dtype=torch.bfloat16 if not CONFIG["use_cpu"] else torch.float32,
            trust_remote_code=True,
            token=CONFIG["hf_token"],
            # Turn off auto device mapping to have explicit control
            device_map=None
        )
        
        # Move model to the specified device
        model = model.to(device)
        load_time = time.time() - load_start_time
        logger.info(f"Model loaded and moved to {device} in {load_time:.2f} seconds")
        log_resource_metrics("base_model_loaded")
            
    except Exception as e:
        logger.warning(f"Failed to load with bfloat16: {e}")
        logger.info("Trying with float32 precision...")
        
        # Try with float32 precision
        load_start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["base_model"],
            trust_remote_code=True,
            token=CONFIG["hf_token"],
            device_map=None
        )
        
        # Move model to the specified device
        model = model.to(device)
        load_time = time.time() - load_start_time
        logger.info(f"Model loaded with float32 and moved to {device} in {load_time:.2f} seconds")
        log_resource_metrics("base_model_loaded")
    
    # Load LoRA weights
    logger.info(f"Loading LoRA weights from: {CONFIG['lora_weights']}")
    lora_start_time = time.time()
    
    # Load the model with LoRA weights
    model = PeftModel.from_pretrained(
        model,
        CONFIG["lora_weights"],
        is_trainable=False
    )
    
    lora_time = time.time() - lora_start_time
    logger.info(f"LoRA weights loaded in {lora_time:.2f} seconds")
    log_resource_metrics("lora_weights_loaded")
    
    # Ensure model is in evaluation mode
    model.eval()
    
    logger.info("Model loaded successfully!")
    
    return model, tokenizer, device

def generate_text(model, tokenizer, prompt, device):
    # Log generation start
    log_resource_metrics(f"generation_start: {prompt[:50]}...")
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Verify model is on the expected device
    device_issues = False
    for name, param in model.named_parameters():
        if param.device != device:
            logger.warning(f"Parameter '{name}' on unexpected device: {param.device} vs expected {device}")
            device_issues = True
            break
    
    if device_issues:
        logger.info(f"Moving entire model to {device} to resolve device mismatches")
        model = model.to(device)
    
    # Generate text
    generation_start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=CONFIG["max_length"],
            do_sample=True,
            temperature=CONFIG["temperature"],
            top_p=CONFIG["top_p"],
        )
    
    generation_time = time.time() - generation_start_time
    
    # Decode generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    tokens_generated = len(outputs[0]) - len(inputs["input_ids"][0])
    tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
    
    logger.info(f"Generated {tokens_generated} tokens in {generation_time:.2f} seconds "
                f"({tokens_per_second:.2f} tokens/sec)")
    
    # Log generation end
    log_resource_metrics(f"generation_end: {tokens_generated} tokens, {generation_time:.2f} seconds")
    
    return generated_text, tokens_generated, generation_time

def main():
    global monitor_resources
    
    print("\n" + "="*50)
    print("Loading fine-tuned Gemma Model with LoRA adapters...")
    print(f"All logs will be stored in: {os.path.abspath(CONFIG['log_dir'])}")
    
    # Start resource monitoring thread
    monitor_thread = threading.Thread(target=resource_monitoring_thread)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    try:
        # Log startup
        log_resource_metrics("application_start")
        
        # Load model with LoRA weights
        model, tokenizer, device = load_model_with_lora()
        
        # Report resource usage after model loading
        log_resource_metrics("model_loaded_complete")
        if CONFIG["print_resources"]:
            print_resource_summary()
        
        # Interactive generation loop
        print("\n" + "="*50)
        print("Fine-tuned Gemma Model with LoRA - Interactive Mode")
        print("Enter 'quit', 'exit', or 'bye' to terminate")
        print("="*50 + "\n")
        
        while True:
            # Get prompt from user
            prompt = input("\nEnter your prompt: ")
            
            # Check if user wants to quit
            if prompt.lower() in ["quit", "exit", "bye", "terminate"]:
                print("\nExiting interactive mode. Goodbye!")
                break
            
            # Generate and display text
            try:
                print("\nGenerating response...\n")
                generated_text, tokens, generation_time = generate_text(model, tokenizer, prompt, device)
                print("="*80)
                print(generated_text)
                print("="*80)
                
                # Print generation stats and resource usage
                print(f"\nGenerated {tokens} tokens in {generation_time:.2f} seconds "
                      f"({tokens/generation_time:.2f} tokens/sec)")
                
                if CONFIG["print_resources"]:
                    print_resource_summary()
                    
            except Exception as e:
                logger.error(f"Error generating text: {e}")
                print(f"Error generating text: {e}")
                log_resource_metrics(f"generation_error: {str(e)[:100]}")
                
                # Try to detect and fix device issues
                if "device" in str(e).lower():
                    print("Attempting to fix device placement issues...")
                    try:
                        # Move the entire model to the device again
                        model = model.to(device)
                        log_resource_metrics("model_device_fix_attempted")
                        print("Model moved to device. Please try again.")
                    except Exception as move_error:
                        logger.error(f"Failed to fix device issues: {move_error}")
                        print(f"Failed to fix device issues: {move_error}")
        
        # Log shutdown
        log_resource_metrics("application_exit")
    
    except Exception as e:
        logger.error(f"Error during model loading: {e}")
        print(f"Error during model loading: {e}")
        log_resource_metrics(f"application_error: {str(e)[:100]}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Stop resource monitoring
    monitor_resources = False
    monitor_thread.join(timeout=2.0)
    print(f"\nResource monitoring stopped. Log file: {os.path.abspath(CONFIG['resource_log'])}")

if __name__ == "__main__":
    main()