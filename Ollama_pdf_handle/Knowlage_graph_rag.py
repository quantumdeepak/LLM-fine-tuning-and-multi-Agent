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
from typing import List, Dict, Any, Tuple, Set, Optional
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import pandas as pd

# Try importing the essentials
try:
    import numpy as np
    from tqdm import tqdm
    import torch
    import ollama
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install the required packages:")
    print("pip install numpy tqdm torch ollama networkx matplotlib pyvis pandas")
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
    "kg_db_path": "knowledge_graph_db",   # Directory to store knowledge graph
    
    # Model settings
    "ollama_model": "gemma3:12b",         # Ollama model to use
    "cuda_device": "3",                   # GPU device to use - CHANGED TO 4
    "temperature": 0.3,                   # Lower temperature for extraction
    "max_tokens": 4096,                   # Maximum tokens in response
    
    # Knowledge Graph settings
    "rebuild_kg": False,                  # Whether to rebuild the KG on startup
    "extraction_batch_size": 5,           # Number of text chunks to process at once
    
    # Chunking parameters
    "chunk_size": 1000,                   # Size of text chunks for extraction
    "chunk_overlap": 200,                 # Overlap between chunks
    "min_chunk_size": 50,                 # Minimum chunk size to keep
    
    # Retrieval parameters
    "top_k": 5,                           # Number of triples to retrieve per query
    
    # Visualization parameters
    "graph_height": "800px",              # Height of the interactive visualization
    "graph_width": "1200px",              # Width of the interactive visualization
    "node_size": 25,                      # Size of nodes in visualization
    "edge_length": 200,                   # Length of edges in visualization
    "node_types": {                       # Node types and their colors
        "entity": "#66CCFF",              # Light blue for entities
        "concept": "#FFCC66",             # Light orange for concepts
        "event": "#99CC99",               # Light green for events
    },
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

# Chunk text for knowledge extraction
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

# Extract triples (subject-predicate-object) from text using LLM
def extract_triples_from_text(text, metadata):
    extraction_prompt = f"""
    I need to extract structured knowledge from the text below. Please identify important entities, concepts, and relationships to build a knowledge graph.

    Extract information in the following JSON format:
    ```json
    [
      {{
        "subject": "entity or concept name",
        "predicate": "relationship",
        "object": "entity or concept name",
        "confidence": 0.95,
        "type": "factual|conceptual|causal"
      }}
    ]
    ```

    Guidelines:
    - Subject and object should be concise noun phrases
    - Predicate should be a clear relationship (e.g., "has_symptom", "causes", "is_part_of")
    - Confidence should be a number between 0 and 1
    - Type can be "factual" (facts), "conceptual" (definitions/categorizations), or "causal" (cause-effect)
    - Extract at least 5-10 high-quality relationships
    - Focus on unique, informative relationships
    - Skip vague or self-evident relationships

    Text to analyze:
    {text}

    Please return ONLY the JSON array with the extracted relationships, no explanations or other text.
    """
    
    try:
        response = ollama.chat(
            model=CONFIG["ollama_model"],
            messages=[{"role": "user", "content": extraction_prompt}],
            options={
                "temperature": CONFIG["temperature"],
                "num_predict": CONFIG["max_tokens"]
            }
        )
        
        response_text = response["message"]["content"]
        
        # Extract JSON from the response
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
        
        # Parse JSON
        try:
            triples = json.loads(json_str)
            
            # Add metadata to each triple
            for triple in triples:
                triple["source"] = metadata["source"]
                triple["chunk_id"] = metadata["id"]
            
            return triples
        except json.JSONDecodeError:
            print(f"Failed to parse JSON: {json_str[:100]}...")
            return []
            
    except Exception as e:
        print(f"Error during triple extraction: {e}")
        return []

# Knowledge Graph implementation
class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.triples = []
        
    def add_triples(self, triples):
        for triple in triples:
            subj = triple["subject"]
            pred = triple["predicate"]
            obj = triple["object"]
            confidence = triple.get("confidence", 0.5)
            rel_type = triple.get("type", "factual")
            source = triple.get("source", "unknown")
            chunk_id = triple.get("chunk_id", "unknown")
            
            # Add nodes if they don't exist
            if not self.graph.has_node(subj):
                self.graph.add_node(subj, type="entity")
                
            if not self.graph.has_node(obj):
                self.graph.add_node(obj, type="entity")
            
            # Add edge with attributes
            self.graph.add_edge(
                subj, 
                obj, 
                predicate=pred,
                weight=confidence,
                type=rel_type,
                source=source,
                chunk_id=chunk_id
            )
            
        # Add to triples list
        self.triples.extend(triples)
    
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
    
    def save(self, filepath):
        data = {
            "graph": nx.node_link_data(self.graph),
            "triples": self.triples
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        kg = cls()
        kg.graph = nx.node_link_graph(data["graph"])
        kg.triples = data["triples"]
        return kg
    
    def get_statistics(self):
        """Return statistics about the knowledge graph"""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_triples": len(self.triples),
            "node_degree": dict(self.graph.degree()),
            "predicates": self.get_predicate_counts(),
            "sources": self.get_source_counts()
        }
    
    def get_predicate_counts(self):
        """Count occurrences of each predicate"""
        predicates = {}
        for _, _, data in self.graph.edges(data=True):
            pred = data.get('predicate', 'unknown')
            predicates[pred] = predicates.get(pred, 0) + 1
        return predicates
    
    def get_source_counts(self):
        """Count triples from each source"""
        sources = {}
        for _, _, data in self.graph.edges(data=True):
            source = data.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        return sources
    
    def visualize(self, output_file="knowledge_graph.html", height="800px", width="1200px"):
        """Generate an interactive visualization of the knowledge graph"""
        # Create a pyvis network
        net = Network(height=height, width=width, notebook=False, directed=True)
        
        # Set physics options for better visualization
        net.barnes_hut(spring_length=CONFIG["edge_length"])
        
        # Add nodes
        for node, attrs in self.graph.nodes(data=True):
            node_type = attrs.get('type', 'entity')
            color = CONFIG["node_types"].get(node_type, "#CCCCCC")
            net.add_node(node, label=node, title=node, color=color, size=CONFIG["node_size"])
        
        # Add edges
        for source, target, data in self.graph.edges(data=True):
            predicate = data.get('predicate', '')
            weight = data.get('weight', 0.5)
            rel_type = data.get('type', 'factual')
            
            # Set edge width based on weight/confidence
            width = 1 + 3 * weight  # Scale from 1-4
            
            # Set edge color based on type
            if rel_type == "factual":
                color = "#0077CC"  # Blue
            elif rel_type == "conceptual":
                color = "#CC7700"  # Orange
            elif rel_type == "causal":
                color = "#00CC77"  # Green
            else:
                color = "#CCCCCC"  # Gray
                
            net.add_edge(source, target, title=predicate, label=predicate, 
                        width=width, color=color)
        
        # Save the visualization
        net.save_graph(output_file)
        print(f"Knowledge graph visualization saved to {output_file}")
        return output_file
    
    def visualize_subgraph(self, query, output_file="subgraph.html"):
        """Visualize a subgraph relevant to a query"""
        # Get relevant triples
        relevant_triples = self.query(query, top_k=CONFIG["top_k"] * 2)
        
        if not relevant_triples:
            print("No relevant information found for visualization")
            return None
        
        # Create subgraph
        subgraph = nx.DiGraph()
        
        # Add nodes and edges from relevant triples
        for triple in relevant_triples:
            subj = triple["subject"]
            pred = triple["predicate"]
            obj = triple["object"]
            relevance = triple.get("relevance", 0.5)
            
            # Add nodes
            if not subgraph.has_node(subj):
                subgraph.add_node(subj)
            if not subgraph.has_node(obj):
                subgraph.add_node(obj)
            
            # Add edge
            subgraph.add_edge(subj, obj, predicate=pred, relevance=relevance)
        
        # Create visualization
        net = Network(height=CONFIG["graph_height"], width=CONFIG["graph_width"], notebook=False, directed=True)
        net.barnes_hut(spring_length=CONFIG["edge_length"])
        
        # Add nodes
        for node in subgraph.nodes():
            net.add_node(node, label=node, title=node, size=CONFIG["node_size"])
        
        # Add edges
        for source, target, data in subgraph.edges(data=True):
            predicate = data.get('predicate', '')
            relevance = data.get('relevance', 0.5)
            
            # Scale width by relevance
            width = 1 + 4 * relevance
            
            net.add_edge(source, target, title=predicate, label=predicate, width=width)
        
        # Save visualization
        net.save_graph(output_file)
        print(f"Query-specific visualization saved to {output_file}")
        return output_file

# Build or load knowledge graph
def setup_knowledge_graph():
    ensure_directory(CONFIG["kg_db_path"])
    kg_file = os.path.join(CONFIG["kg_db_path"], "knowledge_graph.pkl")
    
    if os.path.exists(kg_file) and not CONFIG["rebuild_kg"]:
        print(f"Loading existing knowledge graph from {kg_file}")
        try:
            return KnowledgeGraph.load(kg_file)
        except Exception as e:
            print(f"Error loading knowledge graph: {e}")
            print("Rebuilding knowledge graph...")
            CONFIG["rebuild_kg"] = True
    
    print("Building new knowledge graph...")
    
    # Create new knowledge graph
    knowledge_graph = KnowledgeGraph()
    
    # Load and process documents
    chunks, metadata = load_and_process_files()
    
    if not chunks:
        print("No documents found or processed. Please add text files to the text directory.")
        sys.exit(1)
    
    print(f"Generated {len(chunks)} chunks from documents")
    
    # Extract triples in batches
    print("Extracting knowledge triples...")
    
    batch_size = CONFIG["extraction_batch_size"]
    for i in tqdm(range(0, len(chunks), batch_size), desc="Extracting triples"):
        end_idx = min(i + batch_size, len(chunks))
        batch_chunks = chunks[i:end_idx]
        batch_metadata = metadata[i:end_idx]
        
        # Process each chunk in the batch
        for chunk, meta in zip(batch_chunks, batch_metadata):
            # Extract triples
            triples = extract_triples_from_text(chunk, meta)
            
            # Add to knowledge graph
            if triples:
                knowledge_graph.add_triples(triples)
                
        # Save intermediate results periodically
        if i % (batch_size * 5) == 0 and i > 0:
            knowledge_graph.save(kg_file)
            print(f"Saved intermediate knowledge graph with {len(knowledge_graph.triples)} triples")
    
    # Save final knowledge graph
    knowledge_graph.save(kg_file)
    print(f"Knowledge graph saved to {kg_file} with {len(knowledge_graph.triples)} triples")
    
    # Generate statistics
    stats = knowledge_graph.get_statistics()
    print(f"Knowledge Graph Statistics:")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Edges: {stats['num_edges']}")
    print(f"  Triples: {stats['num_triples']}")
    print(f"  Top predicates: {sorted(stats['predicates'].items(), key=lambda x: x[1], reverse=True)[:5]}")
    
    return knowledge_graph

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
    
    # Setup knowledge graph
    knowledge_graph = setup_knowledge_graph()
    
    # Generate initial visualization
    viz_file = knowledge_graph.visualize()
    print(f"Initial knowledge graph visualization created: {viz_file}")
    
    print("\nKnowledge Graph RAG-powered Chat with", CONFIG["ollama_model"])
    print("Type 'quit', 'exit', 'bye', 'end', or 'terminate' to end the chat")
    print("Type 'rebuild' to rebuild the knowledge graph")
    print("Type 'stats' to show knowledge graph statistics")
    print("Type 'viz' to visualize the full knowledge graph")
    print("Type 'viz:query' to visualize a subgraph relevant to your query")
    print("Type 'add' to manually add knowledge to the graph")
    print("Type 'config' to show current configuration")
    print("="*50)
    
    while True:
        query = input("\nYou: ")
        query = query.strip()
        
        if query.lower() in ["quit", "exit", "bye", "end", "terminate"]:
            print("Ending chat session. Goodbye!")
            break
            
        elif query.lower() == "rebuild":
            print("Rebuilding knowledge graph...")
            CONFIG["rebuild_kg"] = True
            knowledge_graph = setup_knowledge_graph()
            continue
            
        elif query.lower() == "stats":
            stats = knowledge_graph.get_statistics()
            print("\nKnowledge Graph Statistics:")
            print(f"  Nodes: {stats['num_nodes']}")
            print(f"  Edges: {stats['num_edges']}")
            print(f"  Triples: {stats['num_triples']}")
            print(f"  Top predicates: {sorted(stats['predicates'].items(), key=lambda x: x[1], reverse=True)[:10]}")
            print(f"  Top sources: {sorted(stats['sources'].items(), key=lambda x: x[1], reverse=True)[:5]}")
            
            # Node degree distribution
            degrees = list(stats['node_degree'].values())
            print("\nNode Degree Distribution:")
            print(f"  Min degree: {min(degrees)}")
            print(f"  Max degree: {max(degrees)}")
            print(f"  Average degree: {sum(degrees)/len(degrees):.2f}")
            continue
            
        elif query.lower() == "viz":
            viz_file = knowledge_graph.visualize()
            print(f"Knowledge graph visualization updated: {viz_file}")
            continue
            
        elif query.lower().startswith("viz:"):
            viz_query = query[4:].strip()
            if viz_query:
                viz_file = knowledge_graph.visualize_subgraph(viz_query)
                if viz_file:
                    print(f"Query-specific visualization created: {viz_file}")
            else:
                print("Please provide a query after 'viz:' for subgraph visualization")
            continue
        
        elif query.lower() == "add":
            knowledge_add_wizard(knowledge_graph)
            continue
            
        elif query.lower() == "config":
            print("\nCurrent Configuration:")
            for key, value in CONFIG.items():
                print(f"{key}: {value}")
            continue
        
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

# Command-line wizard for adding knowledge to the graph
def knowledge_add_wizard(knowledge_graph):
    print("\nKnowledge Graph Addition Wizard")
    print("="*50)
    print("Enter triples to add to the knowledge graph (or 'done' to finish)")
    
    while True:
        print("\nNew Triple:")
        subject = input("Subject (entity/concept): ").strip()
        
        if subject.lower() == 'done':
            break
            
        predicate = input("Predicate (relationship): ").strip()
        object_entity = input("Object (entity/concept): ").strip()
        
        type_options = ['factual', 'conceptual', 'causal']
        print("Type options:", ", ".join(type_options))
        triple_type = input("Type [factual]: ").strip().lower() or 'factual'
        
        if triple_type not in type_options:
            print(f"Invalid type. Using 'factual' instead.")
            triple_type = 'factual'
            
        confidence = input("Confidence (0.0-1.0) [0.9]: ").strip() or '0.9'
        try:
            confidence = float(confidence)
            if confidence < 0 or confidence > 1:
                print("Confidence must be between 0 and 1. Using 0.9 instead.")
                confidence = 0.9
        except ValueError:
            print("Invalid confidence value. Using 0.9 instead.")
            confidence = 0.9
            
        source = input("Source [manual]: ").strip() or 'manual'
        
        triple = {
            "subject": subject,
            "predicate": predicate,
            "object": object_entity,
            "confidence": confidence,
            "type": triple_type,
            "source": source,
            "chunk_id": "manual"
        }
        
        # Add to knowledge graph
        knowledge_graph.add_triples([triple])
        print(f"Added: {subject} {predicate} {object_entity}")
    
    # Save updated knowledge graph
    kg_file = os.path.join(CONFIG["kg_db_path"], "knowledge_graph.pkl")
    knowledge_graph.save(kg_file)
    print(f"Knowledge graph updated and saved with {len(knowledge_graph.triples)} total triples")

# Main function that directly provides all functionality
def main():
    # Create necessary directories
    ensure_directory(CONFIG["text_directory"])  # This creates the directory for text files if it doesn't exist
    ensure_directory(CONFIG["kg_db_path"])      # This creates the directory for knowledge graph storage if it doesn't exist
    
    # UNCOMMENT ONE OF THESE LINES:
    
    setup_knowledge_graph()  # This builds the knowledge graph only (no chat interface)
    chat_with_kg()           # This starts the chat interface with the knowledge graph
    
    # NOTE: The chat interface already includes all functionality:
    # - 'viz' command generates full knowledge graph visualization
    # - 'viz:query' generates visualization of subgraph relevant to a query
    # - 'stats' shows detailed knowledge graph statistics
    # - 'rebuild' rebuilds the knowledge graph
    # - 'add' allows manual addition of knowledge to the graph

if __name__ == "__main__":
    main()