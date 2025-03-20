import os
import sys
import glob
import re
import time
import json
import pickle
import traceback
import signal
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Try importing the essentials
try:
    import numpy as np
    from tqdm import tqdm
    import torch
    from transformers import AutoTokenizer, AutoModel
    import ollama
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install the required packages:")
    print("pip install numpy tqdm torch transformers ollama")
    sys.exit(1)

# Handle graceful exit
def signal_handler(sig, frame):
    print("\nInterrupted by user. Exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Configuration
CONFIG = {
    # File and directory paths
    "text_directory": "/mnt/DATA/Glucoma/LLM/Ollama_pdf_handle/Text_pdf",  # Directory with text files
    "vector_db_path": "vector_db",        # Directory to store vector database
    
    # Model settings
    "ollama_model": "gemma3:12b",         # Ollama model to use
    "cuda_device": "3",                   # GPU device to use
    "temperature": 0.7,                   # Temperature for text generation
    "max_tokens": 4096,                   # Maximum tokens in response
    
    # Embedding settings
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",  # HF model for embeddings
    "embedding_dimension": 384,           # Embedding dimension for the model
    "rebuild_db": False,                  # Whether to rebuild the vector DB on startup
    
    # Chunking parameters
    "chunk_size": 1000,                   # Size of text chunks for embedding
    "chunk_overlap": 200,                 # Overlap between chunks
    "min_chunk_size": 50,                 # Minimum chunk size to keep
    
    # Retrieval parameters
    "top_k": 5,                           # Number of documents to retrieve per query
    
    # Processing parameters
    "batch_size": 32,                     # Batch size for embedding generation
}

# Create directory if it doesn't exist
def ensure_directory(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)

# Configure GPU
def configure_gpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_device"]
    
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("No CUDA GPUs found, falling back to CPU")
        return False

# Mean Pooling for sentence embeddings
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Initialize embedding model
def initialize_embedding_model():
    print(f"Loading embedding model: {CONFIG['embedding_model']}")
    
    try:
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(CONFIG['embedding_model'])
        model = AutoModel.from_pretrained(CONFIG['embedding_model'])
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        if device.type == 'cuda':
            print(f"Embedding model loaded on GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Embedding model loaded on CPU")
        
        # Create embedding function
        def get_embeddings(texts):
            if isinstance(texts, str):
                texts = [texts]
                
            # Process in batches
            all_embeddings = []
            batch_size = CONFIG['batch_size']
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                encoded_input = tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors='pt'
                )
                
                # Move to device
                encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
                
                # Compute embeddings
                with torch.no_grad():
                    model_output = model(**encoded_input)
                
                # Pool embeddings
                batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                
                # Normalize
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                
                # Move to CPU and convert to numpy
                batch_embeddings = batch_embeddings.cpu().numpy()
                all_embeddings.append(batch_embeddings)
            
            # Combine batches
            if len(all_embeddings) == 1 and len(texts) == 1:
                return all_embeddings[0][0]  # Return single embedding
            return np.vstack(all_embeddings)
        
        return get_embeddings
    
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        traceback.print_exc()
        sys.exit(1)

# Chunk text for embedding
def chunk_text(text):
    if not text:
        return []
    
    # Split by paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding paragraph exceeds chunk size and we have content, 
        # save current chunk and start a new one
        if len(current_chunk) + len(paragraph) > CONFIG["chunk_size"] and current_chunk:
            chunks.append(current_chunk)
            # Keep overlap for context
            current_chunk = current_chunk[-CONFIG["chunk_overlap"]:] if CONFIG["chunk_overlap"] > 0 else ""
        
        # Add paragraph
        if current_chunk:
            current_chunk += "\n\n" + paragraph
        else:
            current_chunk = paragraph
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    # Handle very large paragraphs
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= CONFIG["chunk_size"]:
            final_chunks.append(chunk)
        else:
            # Split by sentence or character
            for i in range(0, len(chunk), CONFIG["chunk_size"] - CONFIG["chunk_overlap"]):
                final_chunks.append(chunk[i:i + CONFIG["chunk_size"]])
    
    # Filter out chunks that are too small
    return [chunk for chunk in final_chunks if len(chunk) >= CONFIG["min_chunk_size"]]

# Load and process text files
def load_and_process_files():
    text_files = glob.glob(os.path.join(CONFIG["text_directory"], "**/*.txt"), recursive=True)
    print(f"Found {len(text_files)} text files")
    
    all_chunks = []
    all_metadata = []
    
    for file_idx, file_path in enumerate(tqdm(text_files, desc="Processing files")):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Chunk the text
            chunks = chunk_text(text)
            
            for chunk_idx, chunk in enumerate(chunks):
                # Add metadata
                rel_path = os.path.relpath(file_path, CONFIG["text_directory"])
                metadata = {
                    "source": rel_path,
                    "file_path": file_path,
                    "chunk_idx": chunk_idx,
                    "file_idx": file_idx,
                    "id": f"doc_{file_idx}_chunk_{chunk_idx}"
                }
                
                all_chunks.append(chunk)
                all_metadata.append(metadata)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return all_chunks, all_metadata

# Simple vector database implementation
class SimpleVectorDB:
    def __init__(self, embedding_dim):
        self.embeddings = None  # Will be numpy array
        self.documents = []     # Original text chunks
        self.metadata = []      # Metadata for each chunk
        self.embedding_dim = embedding_dim
    
    def add_documents(self, documents, metadata, embeddings):
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        self.documents.extend(documents)
        self.metadata.extend(metadata)
    
    def similarity_search(self, query_embedding, top_k=5):
        # Compute cosine similarity
        dot_product = np.dot(self.embeddings, query_embedding)
        
        # Get top k results
        indices = np.argsort(dot_product)[-top_k:][::-1]
        similarities = dot_product[indices]
        
        results = []
        for i, idx in enumerate(indices):
            results.append({
                "document": self.documents[idx],
                "metadata": self.metadata[idx],
                "score": float(similarities[i])
            })
        
        return results
    
    def save(self, filepath):
        data = {
            "embeddings": self.embeddings,
            "documents": self.documents,
            "metadata": self.metadata,
            "embedding_dim": self.embedding_dim
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        db = cls(data["embedding_dim"])
        db.embeddings = data["embeddings"]
        db.documents = data["documents"]
        db.metadata = data["metadata"]
        return db

# Build or load vector database
def setup_vector_db(embedding_function):
    ensure_directory(CONFIG["vector_db_path"])
    db_file = os.path.join(CONFIG["vector_db_path"], "vector_db.pkl")
    
    if os.path.exists(db_file) and not CONFIG["rebuild_db"]:
        print(f"Loading existing vector database from {db_file}")
        try:
            return SimpleVectorDB.load(db_file)
        except Exception as e:
            print(f"Error loading database: {e}")
            print("Rebuilding database...")
            CONFIG["rebuild_db"] = True
    
    print("Building new vector database...")
    
    # Create new database
    vector_db = SimpleVectorDB(CONFIG["embedding_dimension"])
    
    # Load and process documents
    chunks, metadata = load_and_process_files()
    
    if not chunks:
        print("No documents found or processed. Please add text files to the text directory.")
        sys.exit(1)
    
    print(f"Generated {len(chunks)} chunks from documents")
    
    # Generate embeddings in batches
    print("Generating embeddings...")
    
    batch_size = CONFIG["batch_size"]
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks"):
        end_idx = min(i + batch_size, len(chunks))
        batch_chunks = chunks[i:end_idx]
        batch_metadata = metadata[i:end_idx]
        
        # Generate embeddings
        batch_embeddings = embedding_function(batch_chunks)
        
        # Add to database
        vector_db.add_documents(batch_chunks, batch_metadata, batch_embeddings)
    
    # Save database
    vector_db.save(db_file)
    print(f"Vector database saved to {db_file}")
    
    return vector_db

# Create RAG prompt
def create_rag_prompt(query, results):
    # Format context from retrieved documents
    context_parts = []
    
    for i, result in enumerate(results):
        source = result["metadata"].get("source", "Unknown")
        score = result["score"]
        similarity_pct = f"{score * 100:.1f}% relevance"
        
        context_parts.append(f"Document {i+1}: {source} ({similarity_pct})\n\n{result['document']}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Create prompt
    prompt = f"""You are a helpful AI assistant with access to the following documents:

{context}

Based on the information above, please answer the following question:
{query}

If the information needed to answer the question isn't in the documents, please say so, but try to be helpful with what you know.
"""
    return prompt

# Chat function
def chat_with_rag():
    # Configure GPU
    configure_gpu()
    
    # Initialize embedding model
    embedding_function = initialize_embedding_model()
    
    # Setup vector database
    vector_db = setup_vector_db(embedding_function)
    
    print("\nRAG-powered Chat with", CONFIG["ollama_model"])
    print("Type 'quit', 'exit', 'bye', 'end', or 'terminate' to end the chat")
    print("Type 'rebuild' to rebuild the vector database")
    print("Type 'config' to show current configuration")
    print("="*50)
    
    while True:
        query = input("\nYou: ")
        query = query.strip()
        
        if query.lower() in ["quit", "exit", "bye", "end", "terminate"]:
            print("Ending chat session. Goodbye!")
            break
            
        elif query.lower() == "rebuild":
            print("Rebuilding vector database...")
            CONFIG["rebuild_db"] = True
            vector_db = setup_vector_db(embedding_function)
            continue
            
        elif query.lower() == "config":
            print("\nCurrent Configuration:")
            for key, value in CONFIG.items():
                print(f"{key}: {value}")
            continue
        
        # Show "thinking" animation
        print("Retrieving relevant information", end="")
        for _ in range(3):
            time.sleep(0.3)
            print(".", end="", flush=True)
        print()
        
        try:
            # Generate query embedding
            query_embedding = embedding_function(query)
            
            # Retrieve similar documents
            results = vector_db.similarity_search(
                query_embedding=query_embedding, 
                top_k=CONFIG["top_k"]
            )
            
            # Display sources
            print(f"Retrieved {len(results)} relevant document chunks:")
            for i, result in enumerate(results):
                source = result["metadata"].get("source", "Unknown")
                score = result["score"]
                preview = result["document"][:150] + "..." if len(result["document"]) > 150 else result["document"]
                print(f"  {i+1}. {source} (score: {score:.4f})")
                print(f"     {preview}\n")
            
            if not results:
                print("No relevant documents found. Answering based on model knowledge only.")
                prompt = query
            else:
                # Create RAG prompt
                prompt = create_rag_prompt(query, results)
            
            # Show "generating" animation
            print("Generating response", end="")
            for _ in range(3):
                time.sleep(0.3)
                print(".", end="", flush=True)
            print("\n")
            
            # Get response from Ollama
            response = ollama.chat(
                model=CONFIG["ollama_model"],
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": CONFIG["temperature"],
                    "num_predict": CONFIG["max_tokens"]
                }
            )
            
            answer = response["message"]["content"]
            print("Bot:", answer)
            
        except Exception as e:
            print(f"Error generating response: {e}")
            traceback.print_exc()
        
        print("\n" + "-"*50)

if __name__ == "__main__":
    ensure_directory(CONFIG["text_directory"])
    ensure_directory(CONFIG["vector_db_path"])
    chat_with_rag()