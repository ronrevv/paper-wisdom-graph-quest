
"""
Knowledge graph construction module for GraphRAG.
This module handles extracting entities and relationships to build a knowledge graph.
"""

import os
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import json
from typing import List, Dict, Any, Tuple, Set, Optional, Union
from collections import defaultdict

class GraphBuilder:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the graph builder with a spaCy model.
        
        Args:
            spacy_model (str): Name of the spaCy model to use
        """
        try:
            self.nlp = spacy.load(spacy_model)
            print(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            print(f"Downloading spaCy model: {spacy_model}")
            spacy.cli.download(spacy_model)
            self.nlp = spacy.load(spacy_model)
        
        self.graph = nx.Graph()
        self.chunk_to_entities = {}
        self.entity_to_chunks = defaultdict(set)
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            List[Dict]: List of extracted entities with type and text
        """
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            # Filter out low-value entity types if needed
            if ent.label_ in ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]:
                continue
                
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        return entities
    
    def add_chunk_to_graph(self, chunk: str, chunk_id: Union[int, str]) -> None:
        """
        Add a single text chunk to the knowledge graph.
        
        Args:
            chunk (str): Text chunk to process
            chunk_id (Union[int, str]): ID for the chunk
        """
        # Extract entities from the chunk
        entities = self.extract_entities(chunk)
        
        # Add entities to graph if they don't exist
        chunk_entities = []
        for entity in entities:
            entity_text = entity["text"]
            entity_type = entity["label"]
            entity_key = f"{entity_text}:{entity_type}"
            chunk_entities.append(entity_key)
            
            # Add entity node to graph if it doesn't exist
            if not self.graph.has_node(entity_key):
                self.graph.add_node(entity_key, 
                                     text=entity_text, 
                                     type=entity_type,
                                     node_type="entity")
            
            # Track which chunks contain this entity
            self.entity_to_chunks[entity_key].add(chunk_id)
        
        # Add the chunk itself as a node
        self.graph.add_node(f"chunk:{chunk_id}", 
                           text=chunk[:100] + "..." if len(chunk) > 100 else chunk,
                           node_type="chunk",
                           chunk_id=chunk_id)
        
        # Connect chunk to all its entities
        for entity_key in chunk_entities:
            self.graph.add_edge(f"chunk:{chunk_id}", entity_key, relation="contains")
        
        # Create connections between entities in the same chunk
        for i in range(len(chunk_entities)):
            for j in range(i+1, len(chunk_entities)):
                entity1 = chunk_entities[i]
                entity2 = chunk_entities[j]
                
                # Add or update edge between co-occurring entities
                if self.graph.has_edge(entity1, entity2):
                    self.graph[entity1][entity2]["weight"] += 1
                else:
                    self.graph.add_edge(entity1, entity2, relation="co-occurs", weight=1)
        
        # Store the entities for this chunk
        self.chunk_to_entities[chunk_id] = chunk_entities
    
    def build_graph_from_chunks(self, chunks: List[str], chunk_ids: Optional[List[Union[int, str]]] = None) -> None:
        """
        Build a knowledge graph from multiple text chunks.
        
        Args:
            chunks (List[str]): List of text chunks
            chunk_ids (List[Union[int, str]], optional): List of IDs for the chunks
        """
        # If chunk_ids not provided, use array indices
        if chunk_ids is None:
            chunk_ids = list(range(len(chunks)))
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            self.add_chunk_to_graph(chunk, chunk_ids[i])
        
        print(f"Graph built with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        
        # Count node types
        entity_nodes = sum(1 for _, attrs in self.graph.nodes(data=True) if attrs.get("node_type") == "entity")
        chunk_nodes = sum(1 for _, attrs in self.graph.nodes(data=True) if attrs.get("node_type") == "chunk")
        print(f"Graph contains {entity_nodes} entity nodes and {chunk_nodes} chunk nodes")
    
    def get_related_chunks(self, seed_chunks: List[Union[int, str]], 
                           max_distance: int = 2, 
                           top_k: int = 5) -> List[Union[int, str]]:
        """
        Find chunks related to the seed chunks via the knowledge graph.
        
        Args:
            seed_chunks (List[Union[int, str]]): Starting chunk IDs
            max_distance (int): Maximum distance in the graph to traverse
            top_k (int): Maximum number of related chunks to return
            
        Returns:
            List[Union[int, str]]: IDs of related chunks
        """
        related_chunks = set()
        
        # For each seed chunk
        for chunk_id in seed_chunks:
            chunk_node = f"chunk:{chunk_id}"
            
            # Check if chunk exists in graph
            if not self.graph.has_node(chunk_node):
                continue
            
            # Get entities in this chunk
            entities_in_chunk = [
                neighbor for neighbor in self.graph.neighbors(chunk_node)
                if self.graph.nodes[neighbor]["node_type"] == "entity"
            ]
            
            # For each entity, find related chunks via the graph
            for entity in entities_in_chunk:
                # Get other chunks containing this entity
                for related_chunk_id in self.entity_to_chunks[entity]:
                    if related_chunk_id not in seed_chunks:
                        related_chunks.add(related_chunk_id)
                
                # Get related entities (with higher co-occurrence weights first)
                related_entities = []
                for related_entity, edge_data in self.graph[entity].items():
                    if self.graph.nodes[related_entity]["node_type"] == "entity":
                        related_entities.append((related_entity, edge_data.get("weight", 1)))
                
                # Sort by weight descending
                related_entities.sort(key=lambda x: x[1], reverse=True)
                
                # Get chunks containing related entities
                for related_entity, _ in related_entities[:max_distance*2]:
                    for related_chunk_id in self.entity_to_chunks[related_entity]:
                        if related_chunk_id not in seed_chunks:
                            related_chunks.add(related_chunk_id)
        
        # Convert to list and limit to top_k
        related_chunks_list = list(related_chunks)[:top_k]
        return related_chunks_list
    
    def visualize_graph(self, output_file: str = "knowledge_graph.png", figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Visualize the knowledge graph and save to a file.
        
        Args:
            output_file (str): Path to save the visualization
            figsize (Tuple[int, int]): Figure size in inches
        """
        plt.figure(figsize=figsize)
        
        # Get node colors based on type
        node_colors = []
        node_sizes = []
        
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get("node_type", "unknown")
            if node_type == "entity":
                entity_type = self.graph.nodes[node].get("type", "MISC")
                # Different colors for different entity types
                color_map = {
                    "PERSON": "skyblue",
                    "ORG": "orange",
                    "GPE": "green",
                    "LOC": "yellow",
                    "PRODUCT": "purple",
                    "EVENT": "pink",
                    "WORK_OF_ART": "cyan",
                    "LAW": "magenta",
                    "LANGUAGE": "lime",
                    "FAC": "brown"
                }
                node_colors.append(color_map.get(entity_type, "gray"))
                node_sizes.append(300)
            elif node_type == "chunk":
                node_colors.append("red")
                node_sizes.append(150)
            else:
                node_colors.append("black")
                node_sizes.append(200)
        
        # Use spring layout for graph visualization
        pos = nx.spring_layout(self.graph)
        
        # Draw nodes and edges
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.7)
        nx.draw_networkx_edges(self.graph, pos, width=1, alpha=0.5)
        
        # Draw labels only for entity nodes (to avoid clutter)
        entity_labels = {node: self.graph.nodes[node]["text"] 
                        for node in self.graph.nodes() 
                        if self.graph.nodes[node].get("node_type") == "entity"}
        nx.draw_networkx_labels(self.graph, pos, labels=entity_labels, font_size=8)
        
        plt.title("Knowledge Graph Visualization")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"Graph visualization saved to {output_file}")
    
    def save_graph(self, file_path: str) -> None:
        """
        Save the knowledge graph to a file.
        
        Args:
            file_path (str): Path to save the graph
        """
        # Convert graph to dictionary format for serialization
        graph_data = {
            "nodes": [
                {
                    "id": node_id,
                    **{k: v for k, v in attrs.items() if isinstance(v, (str, int, float, bool))}
                }
                for node_id, attrs in self.graph.nodes(data=True)
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    **{k: v for k, v in attrs.items() if isinstance(v, (str, int, float, bool))}
                }
                for u, v, attrs in self.graph.edges(data=True)
            ],
            "chunk_to_entities": {
                str(chunk_id): entities 
                for chunk_id, entities in self.chunk_to_entities.items()
            },
            "entity_to_chunks": {
                entity: list(chunks)
                for entity, chunks in self.entity_to_chunks.items()
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        print(f"Graph saved to {file_path}")
    
    def load_graph(self, file_path: str) -> None:
        """
        Load a knowledge graph from a file.
        
        Args:
            file_path (str): Path to the saved graph
        """
        with open(file_path, 'r') as f:
            graph_data = json.load(f)
        
        # Create a new graph
        self.graph = nx.Graph()
        
        # Add nodes
        for node_data in graph_data["nodes"]:
            node_id = node_data.pop("id")
            self.graph.add_node(node_id, **node_data)
        
        # Add edges
        for edge_data in graph_data["edges"]:
            source = edge_data.pop("source")
            target = edge_data.pop("target")
            self.graph.add_edge(source, target, **edge_data)
        
        # Restore mappings
        self.chunk_to_entities = {
            int(chunk_id) if chunk_id.isdigit() else chunk_id: entities
            for chunk_id, entities in graph_data["chunk_to_entities"].items()
        }
        
        self.entity_to_chunks = defaultdict(set)
        for entity, chunks in graph_data["entity_to_chunks"].items():
            self.entity_to_chunks[entity] = set(
                int(chunk) if isinstance(chunk, str) and chunk.isdigit() else chunk
                for chunk in chunks
            )
        
        print(f"Graph loaded from {file_path} with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
