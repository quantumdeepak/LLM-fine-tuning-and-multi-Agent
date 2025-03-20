
# Gemma 3 PT Fine-tuning Pipeline for Computer Vision Papers

This project provides a modular pipeline for fine-tuning the Google Gemma 3 image-text-to-text model (`gemma-3-4b-pt`) on computer vision research papers from CVPR.

## Overview

The pipeline enables the model to understand and generate text about computer vision concepts based on visual diagrams, figures, and illustrations from academic papers. It's divided into five independent modules:

1. **Content Extraction**: Extract text and images from PDF papers
2. **Data Preparation**: Process and format data for the image-text model
3. **Model Setup**: Configure Gemma 3 with parameter-efficient LoRA fine-tuning
4. **Training**: Fine-tune the model on academic paper images and text
5. **Testing**: Evaluate the fine-tuned model on new images

## Key Features

- Extracts both **text and images** from CVPR papers
- Identifies **figures and captions** for high-quality training data
- Uses **parameter-efficient fine-tuning** (LoRA) to reduce computational requirements
- Supports multiple types of **visual-language tasks** (caption generation, technical explanation, etc.)
- **Modular architecture** allowing for independent execution of each stage

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gemma-cvpr-finetuning.git
cd gemma-cvpr-finetuning

# Install dependencies
pip install -r requirements.txt

# Note: You'll need Hugging Face access to google/gemma-3-4b-pt
# Follow instructions at huggingface.co to set up proper authentication
```

## Usage

### Running the entire pipeline

```bash
python main.py --pdf_dir /path/to/cvpr_papers --output_dir ./my_finetuned_model
```

### Running specific steps

```bash
# Extract content from PDFs
python main.py --steps extract

# Prepare dataset and train model
python main.py --steps prepare train

# Test the model with a specific image
python main.py --steps test --test_image ./path/to/test_image.jpg
```

### Running individual modules directly

Each module can also be executed independently:

```bash
# Extract content from PDFs
python extract_content.py

# Set up the model
python setup_model.py

# Test the model
python test_model.py
```

## Configuration

All configuration parameters are centralized in `config.py`. Key settings you may want to adjust:

- `MODEL_NAME`: The pre-trained model identifier
- `BATCH_SIZE`: Number of examples per training batch
- `NUM_EPOCHS`: Number of training epochs
- `LEARNING_RATE`: Learning rate for fine-tuning
- `LORA_*`: Parameters for LoRA fine-tuning
- `MAX_LENGTH`: Maximum sequence length for tokenization

## File Structure

```
├── config.py              # Central configuration file
├── extract_content.py     # PDF text and image extraction module
├── prepare_data.py        # Data processing and preparation
├── setup_model.py         # Model download and LoRA configuration
├── train_model.py         # Model fine-tuning
├── test_model.py          # Model evaluation
├── main.py                # Main pipeline script
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

## Model Capabilities

After fine-tuning, the model should be able to:

1. **Describe figures** from computer vision papers
2. **Explain technical diagrams** in the context of the paper
3. **Analyze architectural components** shown in figures
4. **Generate technical descriptions** of visual concepts

## Performance Considerations

- For best results, a GPU with at least 16GB VRAM is recommended
- Processing PDFs can be memory-intensive; consider batching large collections
- LoRA significantly reduces memory requirements compared to full fine-tuning
- Reduce batch size if encountering out-of-memory errors

## Limitations

- The fine-tuned model inherits limitations from the base Gemma 3 model
- Quality of results depends on the extracted images and captions
- Not all figures in papers have explicit captions, which may affect training data quality

## License

This project is released under the same license terms as the Gemma model. See the [Gemma license](https://ai.google.dev/gemma/terms) for details.