# prepare_data.py - Prepare data for image-text-to-text model

import json
import os
import random
import sys
from tqdm import tqdm
# Import the Hugging Face datasets explicitly to avoid conflicts
from huggingface_datasets import Dataset
from transformers import AutoProcessor
from PIL import Image
from config import MODEL_NAME, DATASET_PATH, MAX_LENGTH, IMAGES_DIR

def load_dataset(dataset_path=DATASET_PATH):
    """Load the extracted dataset."""
    print(f"Loading dataset from {dataset_path}...")
    
    try:
        if not os.path.exists(dataset_path):
            print(f"ERROR: Dataset file '{dataset_path}' not found!")
            return []
        
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"Loaded dataset with {len(data)} papers.")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def create_dummy_examples():
    """Create dummy examples for testing when no real data is available."""
    print("Creating dummy examples for testing...")
    
    # Create a dummy image
    dummy_img_dir = os.path.join(IMAGES_DIR, "dummy")
    os.makedirs(dummy_img_dir, exist_ok=True)
    dummy_img_path = os.path.join(dummy_img_dir, "dummy_image.jpg")
    
    # Create a simple image with PIL if it doesn't exist
    if not os.path.exists(dummy_img_path):
        try:
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (300, 300), color=(73, 109, 137))
            d = ImageDraw.Draw(img)
            d.rectangle([(50, 50), (250, 250)], outline=(255, 255, 255))
            d.ellipse([(100, 100), (200, 200)], fill=(255, 0, 0))
            img.save(dummy_img_path)
            print(f"Created dummy test image at {dummy_img_path}")
        except Exception as e:
            print(f"Could not create dummy image: {e}")
            # If we can't create an image, we can't proceed
            print("Cannot proceed without any images. Exiting.")
            sys.exit(1)
    
    # Create dummy examples
    dummy_examples = [
        {
            "paper_id": "dummy_paper",
            "image_path": dummy_img_path,
            "prompt": "Describe this figure:",
            "target": "This figure shows a neural network architecture for object detection.",
            "type": "caption_generation"
        },
        {
            "paper_id": "dummy_paper",
            "image_path": dummy_img_path,
            "prompt": "Explain the technical details of this diagram:",
            "target": "The diagram illustrates a CNN-based architecture with multiple convolutional layers followed by fully connected layers.",
            "type": "technical_analysis"
        }
    ]
    
    print(f"Created {len(dummy_examples)} dummy examples for testing.")
    return dummy_examples

def create_training_examples(data, processor):
    """Create training examples for the image-text-to-text model."""
    examples = []
    
    # If data is empty, create dummy examples
    if not data:
        return create_dummy_examples()
    
    for paper in tqdm(data, desc="Creating training examples"):
        # Skip papers without images
        if not paper.get("images"):
            continue
        
        paper_id = paper["paper_id"]
        sections = paper.get("sections", {})
        images = paper["images"]
        
        # Process only figures with captions
        for img_data in images:
            if not img_data.get("is_figure") or not img_data.get("caption"):
                continue
            
            image_path = img_data["image_path"]
            caption = img_data["caption"]
            
            # Skip if image file doesn't exist
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found.")
                continue
            
            try:
                # Verify image can be opened
                Image.open(image_path).convert("RGB")
                
                # Create different types of training examples for this image-caption pair
                
                # 1. Caption generation
                examples.append({
                    "paper_id": paper_id,
                    "image_path": image_path,
                    "prompt": "Describe this figure:",
                    "target": caption,
                    "type": "caption_generation"
                })
                
                # 2. Figure explanation with context
                # Get context from methodology or related section
                context = sections.get("methodology", "") or sections.get("approach", "")
                if context:
                    # Truncate context if too long
                    context = context[:500] + "..." if len(context) > 500 else context
                    
                    examples.append({
                        "paper_id": paper_id,
                        "image_path": image_path,
                        "prompt": f"Explain this figure in the context of the following methodology:\n{context}",
                        "target": caption,
                        "type": "figure_explanation"
                    })
                
                # 3. Technical analysis
                examples.append({
                    "paper_id": paper_id,
                    "image_path": image_path,
                    "prompt": "Analyze the technical components shown in this figure:",
                    "target": caption,
                    "type": "technical_analysis"
                })
                
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
    
    if not examples:
        print("No valid examples created from the dataset. Creating dummy examples instead.")
        return create_dummy_examples()
    
    print(f"Created {len(examples)} training examples from {len(data)} papers")
    return examples

def prepare_data_for_training(examples, processor):
    """Convert raw examples to model inputs."""
    processed_examples = []
    
    for example in tqdm(examples, desc="Processing examples"):
        try:
            # Load image
            image = Image.open(example["image_path"]).convert("RGB")
            
            # Process inputs
            inputs = processor(
                text=example["prompt"],
                images=image,
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )
            
            # Process targets
            target_encoding = processor(
                text=example["target"],
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )
            
            # Prepare example with all required fields
            processed_example = {
                "paper_id": example["paper_id"],
                "type": example["type"],
                "pixel_values": inputs.pixel_values[0],
                "input_ids": inputs.input_ids[0],
                "attention_mask": inputs.attention_mask[0],
                "labels": target_encoding.input_ids[0]
            }
            
            processed_examples.append(processed_example)
            
        except Exception as e:
            print(f"Error processing example: {e}")
    
    return processed_examples

def prepare_data():
    """Load, process, and split the dataset."""
    print("Preparing datasets for training...")
    
    # Load processor
    print(f"Loading processor from {MODEL_NAME}...")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"Error loading processor: {e}")
        print("This could be due to authentication issues with the model or network problems.")
        print("Please ensure you have proper access to the Gemma model.")
        sys.exit(1)
    
    # Load raw data
    data = load_dataset()
    
    # Create examples
    examples = create_training_examples(data, processor)
    
    # Process examples
    processed_examples = prepare_data_for_training(examples, processor)
    
    if not processed_examples:
        print("No processed examples available. Cannot proceed with training.")
        sys.exit(1)
    
    # Create datasets
    dataset = Dataset.from_list(processed_examples)
    
    # Split into train and evaluation datasets
    print("Splitting dataset into train and evaluation sets...")
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"Train dataset: {len(train_dataset)} examples")
    print(f"Evaluation dataset: {len(eval_dataset)} examples")
    
    return train_dataset, eval_dataset, processor

if __name__ == "__main__":
    prepare_data()