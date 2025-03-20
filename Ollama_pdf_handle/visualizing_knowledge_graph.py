import os
import pickle
import networkx as nx
from pyvis.network import Network
import random
from pathlib import Path

# Configuration
KG_DB_PATH = "knowledge_graph_db"  # Path to knowledge graph directory
OUTPUT_DIR = "kg_visualization"     # Directory to save visualization files

def load_knowledge_graph(kg_file):
    """Load the knowledge graph from the pickle file"""
    try:
        with open(kg_file, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict) and "graph" in data:
            graph = nx.node_link_graph(data["graph"])
            print(f"Loaded knowledge graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            return graph
        else:
            print("Error: Unexpected knowledge graph format")
            return nx.DiGraph()
            
    except Exception as e:
        print(f"Error loading knowledge graph: {e}")
        return nx.DiGraph()

def create_visualization(graph, output_file):
    """Create a standalone HTML visualization of the graph"""
    # Create a pyvis network
    net = Network(
        height="800px", 
        width="100%", 
        directed=True,
        notebook=False,
        cdn_resources="in_line"  # Include all resources in the HTML file
    )
    
    # Configure physics for better visualization
    net.barnes_hut(
        gravity=-2000,
        central_gravity=0.3,
        spring_length=150,
        spring_strength=0.05,
        damping=0.09,
        overlap=0
    )
    
    # Set visualization options
    net.set_options("""
    {
      "nodes": {
        "font": {
          "size": 14,
          "face": "Tahoma"
        }
      },
      "edges": {
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 0.5
          }
        },
        "color": {
          "inherit": true
        },
        "smooth": {
          "enabled": true,
          "type": "dynamic"
        },
        "font": {
          "size": 12,
          "align": "middle"
        }
      },
      "physics": {
        "enabled": true,
        "stabilization": {
          "enabled": true,
          "iterations": 1000,
          "updateInterval": 100
        }
      },
      "interaction": {
        "hover": true,
        "multiselect": true,
        "navigationButtons": true,
        "keyboard": {
          "enabled": true
        }
      }
    }
    """)
    
    # Calculate node sizes based on degree
    degrees = dict(graph.degree())
    max_degree = max(degrees.values()) if degrees else 1
    
    # Add nodes to the network
    for node, attrs in graph.nodes(data=True):
        size = (degrees[node] / max_degree) * 15 + 10  # Scale node size by degree
        
        # Generate consistent node colors based on node type
        node_type = attrs.get('type', 'entity')
        if node_type == 'entity':
            color = '#6FA8DC'  # Blue for entities
        else:
            # Generate deterministic color based on node name
            color_seed = sum(ord(c) for c in str(node)) % 1000
            r = (color_seed * 123) % 255
            g = (color_seed * 45) % 255
            b = (color_seed * 67) % 255
            color = f'rgb({r},{g},{b})'
        
        net.add_node(
            node, 
            label=str(node),
            title=str(node),
            size=size,
            color=color
        )
    
    # Add edges to the network
    for source, target, data in graph.edges(data=True):
        # Extract edge attributes
        predicate = data.get('predicate', '')
        weight = data.get('weight', 0.5)
        rel_type = data.get('type', 'factual')
        source_file = data.get('source', 'unknown')
        
        # Determine edge color based on relationship type
        if rel_type == 'factual':
            color = '#4CAF50'  # Green for factual
        elif rel_type == 'conceptual':
            color = '#2196F3'  # Blue for conceptual
        elif rel_type == 'causal':
            color = '#FF5722'  # Orange for causal
        else:
            color = '#9E9E9E'  # Grey for other types
        
        # Edge width based on confidence/weight
        width = weight * 2 + 1
        
        # Create edge title with detailed information
        title = f"Relation: {predicate}<br>Type: {rel_type}<br>Confidence: {weight:.2f}<br>Source: {source_file}"
        
        net.add_edge(
            source, 
            target, 
            title=title,
            label=predicate,
            width=width,
            color=color
        )
    
    # Save the network to an HTML file
    net.save_graph(output_file)
    
    # Enhance the HTML with additional controls
    enhance_html_file(output_file)
    
    print(f"Visualization saved to: {output_file}")

def enhance_html_file(html_file):
    """Add custom controls and styling to the HTML file"""
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Add custom CSS and controls
        custom_controls = """
        <style>
            body, html {
                margin: 0;
                padding: 0;
                height: 100%;
                width: 100%;
                font-family: Arial, sans-serif;
            }
            #mynetwork {
                width: 100%;
                height: 100vh;
                border: none;
            }
            #controls {
                position: absolute;
                top: 10px;
                left: 10px;
                z-index: 999;
                background-color: rgba(255, 255, 255, 0.8);
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }
            #search-panel {
                position: absolute;
                top: 10px;
                right: 10px;
                z-index: 999;
                background-color: rgba(255, 255, 255, 0.8);
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                width: 200px;
            }
            #info-panel {
                position: absolute;
                bottom: 10px;
                left: 10px;
                z-index: 999;
                background-color: rgba(255, 255, 255, 0.8);
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                max-width: 300px;
            }
            #legend {
                position: absolute;
                bottom: 10px;
                right: 10px;
                z-index: 999;
                background-color: rgba(255, 255, 255, 0.8);
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }
            .legend-item {
                display: flex;
                align-items: center;
                margin-bottom: 5px;
            }
            .legend-color {
                width: 15px;
                height: 15px;
                margin-right: 5px;
                border-radius: 2px;
            }
            button {
                margin: 2px;
                padding: 5px 10px;
                cursor: pointer;
            }
            input[type="text"] {
                width: 100%;
                padding: 5px;
                box-sizing: border-box;
                margin-bottom: 5px;
            }
            #search-results {
                max-height: 200px;
                overflow-y: auto;
            }
            .search-result {
                padding: 5px;
                cursor: pointer;
                border-bottom: 1px solid #eee;
            }
            .search-result:hover {
                background-color: #f0f0f0;
            }
        </style>
        
        <div id="controls">
            <button onclick="zoomIn()">Zoom In</button>
            <button onclick="zoomOut()">Zoom Out</button>
            <button onclick="resetView()">Reset View</button>
            <button onclick="togglePhysics()">Toggle Physics</button>
            <button onclick="exportGraph()">Export PNG</button>
        </div>
        
        <div id="search-panel">
            <input type="text" id="search-input" placeholder="Search nodes..." oninput="searchNodes()">
            <div id="search-results"></div>
        </div>
        
        <div id="info-panel">
            <h3>Selected Node/Edge Info</h3>
            <div id="selection-info">Select a node or edge to see details</div>
        </div>
        
        <div id="legend">
            <h3>Legend</h3>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #6FA8DC;"></div>
                <span>Entity</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #4CAF50;"></div>
                <span>Factual Relation</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #2196F3;"></div>
                <span>Conceptual Relation</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #FF5722;"></div>
                <span>Causal Relation</span>
            </div>
        </div>
        
        <script>
            // Wait for network to be defined
            document.addEventListener("DOMContentLoaded", function() {
                // Attach selection event
                network.on("selectNode", function(params) {
                    if (params.nodes.length > 0) {
                        var nodeId = params.nodes[0];
                        var node = network.body.nodes[nodeId];
                        document.getElementById("selection-info").innerHTML = 
                            "<strong>Node:</strong> " + node.options.label + 
                            "<br><strong>Connections:</strong> " + 
                            network.getConnectedNodes(nodeId).length;
                    }
                });
                
                network.on("selectEdge", function(params) {
                    if (params.edges.length > 0) {
                        var edgeId = params.edges[0];
                        var edge = network.body.edges[edgeId];
                        document.getElementById("selection-info").innerHTML = 
                            "<strong>Relation:</strong> " + edge.options.label + 
                            "<br><strong>From:</strong> " + edge.from.options.label +
                            "<br><strong>To:</strong> " + edge.to.options.label;
                    }
                });
                
                network.on("deselectNode", function(params) {
                    document.getElementById("selection-info").innerHTML = 
                        "Select a node or edge to see details";
                });
                
                network.on("deselectEdge", function(params) {
                    document.getElementById("selection-info").innerHTML = 
                        "Select a node or edge to see details";
                });
            });
            
            function zoomIn() {
                network.zoomIn(0.2);
            }
            
            function zoomOut() {
                network.zoomOut(0.2);
            }
            
            function resetView() {
                network.fit({
                    animation: {
                        duration: 1000,
                        easingFunction: "easeInOutQuad"
                    }
                });
            }
            
            function togglePhysics() {
                var physics = network.physics.options.enabled;
                network.setOptions({ physics: { enabled: !physics } });
            }
            
            function exportGraph() {
                var canvas = document.getElementsByTagName("canvas")[0];
                var link = document.createElement('a');
                link.href = canvas.toDataURL("image/png");
                link.download = 'knowledge_graph.png';
                link.click();
            }
            
            function searchNodes() {
                var searchTerm = document.getElementById("search-input").value.toLowerCase();
                var resultsDiv = document.getElementById("search-results");
                resultsDiv.innerHTML = "";
                
                if (searchTerm.length < 2) return;
                
                var nodes = network.body.nodes;
                var results = [];
                
                for (var nodeId in nodes) {
                    var node = nodes[nodeId];
                    if (node.options && node.options.label) {
                        var label = node.options.label.toLowerCase();
                        
                        if (label.includes(searchTerm)) {
                            results.push({
                                id: nodeId,
                                label: node.options.label
                            });
                        }
                    }
                }
                
                results.sort((a, b) => a.label.localeCompare(b.label));
                
                for (var i = 0; i < Math.min(results.length, 10); i++) {
                    var div = document.createElement("div");
                    div.className = "search-result";
                    div.textContent = results[i].label;
                    div.setAttribute("data-node-id", results[i].id);
                    div.onclick = function() {
                        var nodeId = this.getAttribute("data-node-id");
                        network.selectNodes([nodeId]);
                        network.focus(nodeId, {
                            scale: 1.2,
                            animation: {
                                duration: 1000,
                                easingFunction: "easeInOutQuad"
                            }
                        });
                    };
                    resultsDiv.appendChild(div);
                }
                
                if (results.length === 0) {
                    resultsDiv.innerHTML = "<div class='search-result'>No results found</div>";
                }
            }
        </script>
        """
        
        # Insert custom controls before the closing body tag
        enhanced_html = html_content.replace('</body>', custom_controls + '</body>')
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(enhanced_html)
            
    except Exception as e:
        print(f"Error enhancing HTML file: {e}")

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Path to knowledge graph file
    kg_file = os.path.join(KG_DB_PATH, "knowledge_graph.pkl")
    
    # Output HTML file path
    output_file = os.path.join(OUTPUT_DIR, "knowledge_graph.html")
    
    # Load knowledge graph
    graph = load_knowledge_graph(kg_file)
    
    if graph.number_of_nodes() > 0:
        # Create visualization
        create_visualization(graph, output_file)
        print(f"\nVisualization complete. Open '{output_file}' in a web browser or with Live Server.")
    else:
        print("Error: No nodes found in the knowledge graph.")

if __name__ == "__main__":
    main()