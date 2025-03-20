# setup_model.py - Download and configure the model using available classes

import os
import torch
from transformers import AutoProcessor, AutoModel
from peft import get_peft_model, LoraConfig, TaskType
from config import MODEL_NAME, LORA_R, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES

def download_model_and_processor():
    """Download the base model and processor."""
    print(f"Downloading model and processor from {MODEL_NAME}...")
    
    try:
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        
        # Use AutoModel instead of AutoModelForImageTextToText
        model = AutoModel.from_pretrained(MODEL_NAME)
        
        print("Model and processor downloaded successfully.")
        return model, processor
    except Exception as e:
        print(f"Error downloading model: {e}")
        print(f"Detailed error information: {str(e)}")
        
        # Try to provide more helpful information about the model
        print("\nTROUBLESHOOTING TIPS:")
        print("1. Check if you're authenticated with Hugging Face for accessing Gemma models")
        print("2. Verify the model name is correct and accessible")
        print("3. Check your internet connection")
        print("4. You might need to use a different model class for your version of transformers")
        
        # Check if we can access any model at all
        try:
            print("\nTrying to load a publicly available model to test connectivity...")
            test_model = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
            print("Successfully loaded test model. The issue is specific to the Gemma model.")
        except Exception as test_e:
            print(f"Failed to load test model: {test_e}")
            print("This suggests a general connectivity or authentication issue.")
        
        raise

def apply_lora(model):
    """Apply LoRA for efficient fine-tuning."""
    print("Applying LoRA for parameter-efficient fine-tuning...")
    
    # Adjust task type based on the model architecture
    task_type = TaskType.SEQ_2_SEQ_LM
    
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=task_type
    )
    
    try:
        peft_model = get_peft_model(model, lora_config)
        
        # Print trainable parameters info
        model_size = sum(p.numel() for p in peft_model.parameters())
        trainable_size = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {model_size:,}")
        print(f"Trainable parameters: {trainable_size:,} ({trainable_size/model_size:.2%})")
        
        return peft_model
    except Exception as e:
        print(f"Error applying LoRA: {e}")
        print("This might indicate that the model architecture is not compatible with the LoRA configuration.")
        print("You may need to adjust the target_modules in config.py to match the actual model architecture.")
        raise

def main():
    """Download model and apply LoRA."""
    model, processor = download_model_and_processor()
    peft_model = apply_lora(model)
    
    # Return them for use by other modules
    return peft_model, processor

if __name__ == "__main__":
    main()