import os
import sys
import re
import json
import pickle
import time
import traceback
import signal
from pathlib import Path
import networkx as nx
import torch
import ollama

# Handle graceful exit
def signal_handler(sig, frame):
    print("\nInterrupted by user. Exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Configuration
CONFIG = {
    # File and directory paths
    "kg_db_path": "knowledge_graph_db",   # Directory where knowledge graph is stored
    
    # Model settings
    "ollama_model": "gemma3:12b",         # Ollama model to use
    "cuda_device": "3",                   # GPU device to use
    "temperature": 0.7,                   # Temperature for generation
    "max_tokens": 4096,                   # Maximum tokens in response
    
    # Retrieval parameters
    "top_k": 5,                           # Number of triples to retrieve per query
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

# Knowledge Graph implementation
class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.triples = []
    
    def query(self, question, top_k=5):
        """
        Query the knowledge graph for relevant information
        Returns top_k most relevant triples
        """
        query_prompt = f"""
        I have a knowledge graph and need to find relevant triples for the following query:
        
        Query: {question}
        
        Please analyze these triples and rank them by relevance to the query from 0 to 1:
        
        {json.dumps(self.triples[:100], indent=2)}
        
        Return ONLY a JSON array of the {top_k} most relevant triples with a new 'relevance' field indicating
        how relevant each triple is to the query (0 to 1).
        """
        
        try:
            response = ollama.chat(
                model=CONFIG["ollama_model"],
                messages=[{"role": "user", "content": query_prompt}],
                options={
                    "temperature": 0.2,
                    "num_predict": CONFIG["max_tokens"]
                }
            )
            
            response_text = response["message"]["content"]
            
            # Extract JSON
            json_pattern = r'```json(.*?)```'
            json_matches = re.findall(json_pattern, response_text, re.DOTALL)
            
            if json_matches:
                json_str = json_matches[0].strip()
            else:
                # Try extracting without code block markers
                json_pattern = r'\[\s*\{.*\}\s*\]'
                json_matches = re.findall(json_pattern, response_text, re.DOTALL)
                
                if json_matches:
                    json_str = json_matches[0].strip()
                else:
                    json_str = response_text.strip()
            
            # Parse the ranked triples
            try:
                ranked_triples = json.loads(json_str)
                return sorted(ranked_triples, key=lambda x: x.get('relevance', 0), reverse=True)[:top_k]
            except json.JSONDecodeError:
                # Fall back to simple keyword matching
                return self.simple_keyword_search(question, top_k)
                
        except Exception as e:
            print(f"Error during knowledge graph query: {e}")
            return self.simple_keyword_search(question, top_k)
    
    def simple_keyword_search(self, query, top_k=5):
        """Fallback method that uses simple keyword matching"""
        query_terms = set(query.lower().split())
        scored_triples = []
        
        for triple in self.triples:
            # Convert all fields to lowercase for comparison
            subj = triple["subject"].lower()
            pred = triple["predicate"].lower()
            obj = triple["object"].lower()
            
            # Count matching terms
            text = f"{subj} {pred} {obj}"
            matching_terms = sum(1 for term in query_terms if term in text)
            
            # Score is proportion of matching terms plus a small factor for confidence
            score = (matching_terms / len(query_terms) if query_terms else 0) 
            score += 0.1 * triple.get("confidence", 0.5)
            
            if score > 0:
                triple_copy = triple.copy()
                triple_copy["relevance"] = min(score, 1.0)  # Cap at 1.0
                scored_triples.append(triple_copy)
        
        # Sort by score and return top_k
        return sorted(scored_triples, key=lambda x: x.get('relevance', 0), reverse=True)[:top_k]
    
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        kg = cls()
        kg.graph = nx.node_link_graph(data["graph"])
        kg.triples = data["triples"]
        return kg

# Create RAG prompt with knowledge graph results
def create_rag_prompt(query, results):
    # Format context from retrieved triples
    context_parts = []
    facts = []
    
    for i, result in enumerate(results):
        subj = result["subject"]
        pred = result["predicate"]
        obj = result["object"]
        source = result.get("source", "Unknown")
        relevance = result.get("relevance", 0)
        
        # Format as natural language fact
        fact = f"{subj} {pred} {obj}"
        facts.append(fact)
        
        # Add to context
        context_parts.append(f"Fact {i+1}: {fact} (Relevance: {relevance:.2f}, Source: {source})")
    
    context = "\n".join(context_parts)
    
    # Create prompt
    prompt = f"""You are a helpful AI assistant with access to the following knowledge graph facts related to the user's question:

{context}

Based on the information above, please answer the following question:
{query}

If the information needed to answer the question isn't in the knowledge graph facts, please say so, but try to be helpful with what you know. Cite the relevant facts from the knowledge graph when appropriate.
"""
    return prompt, facts

# Chat function with knowledge graph
def chat_with_kg():
    # Configure GPU
    configure_gpu()
    
    # Load knowledge graph
    kg_file = os.path.join(CONFIG["kg_db_path"], "knowledge_graph.pkl")
    
    if not os.path.exists(kg_file):
        print(f"Error: Knowledge graph not found at {kg_file}")
        print("Please run build_knowledge_graph.py first to create the knowledge graph.")
        sys.exit(1)
    
    print(f"Loading knowledge graph from {kg_file}...")
    knowledge_graph = KnowledgeGraph.load(kg_file)
    print(f"Loaded knowledge graph with {len(knowledge_graph.triples)} triples")
    
    print("\nKnowledge Graph RAG-powered Chat with", CONFIG["ollama_model"])
    print("Type 'quit', 'exit', 'bye', 'end', or 'terminate' to end the chat")
    print("="*50)
    
    while True:
        query = input("\nYou: ")
        query = query.strip()
        
        if query.lower() in ["quit", "exit", "bye", "end", "terminate"]:
            print("Ending chat session. Goodbye!")
            break
        
        # Show "thinking" animation
        print("Retrieving relevant knowledge", end="")
        for _ in range(3):
            time.sleep(0.3)
            print(".", end="", flush=True)
        print()
        
        try:
            # Retrieve relevant triples from knowledge graph
            results = knowledge_graph.query(
                question=query, 
                top_k=CONFIG["top_k"]
            )
            
            # Display sources
            print(f"Retrieved {len(results)} relevant knowledge graph triples:")
            for i, result in enumerate(results):
                subj = result["subject"]
                pred = result["predicate"]
                obj = result["object"]
                relevance = result.get("relevance", 0)
                
                print(f"  {i+1}. {subj} {pred} {obj} (Relevance: {relevance:.2f})")
            
            if not results:
                print("No relevant knowledge found. Answering based on model knowledge only.")
                prompt = query
                facts = []
            else:
                # Create RAG prompt
                prompt, facts = create_rag_prompt(query, results)
            
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
    ensure_directory(CONFIG["kg_db_path"])
    chat_with_kg()