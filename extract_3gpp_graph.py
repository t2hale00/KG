import google.generativeai as genai
from typing import Dict, List, Any, Tuple
import json
import os
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
from preprocess_pdfs import read_pdfs_from_directory, clean_text
from datetime import datetime
import concurrent.futures
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from dotenv import load_dotenv
import time
import re
import hashlib
import random

# Load environment variables from .env file
load_dotenv()

@dataclass
class Node:
    id: str
    type: str
    name: str
    properties: Dict[str, Any]

@dataclass
class Edge:
    source: str
    target: str
    type: str
    properties: Dict[str, Any]

class ThreeGPPGraphExtractor:
    def __init__(self, api_key: str = None, batch_size: int = 5):
        """Initialize the extractor with Google AI API."""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required")
        
        # Configure Gemini API
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Set up generation config for precise extraction
        self.generation_config = {
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048
        }
        
        # Batch processing settings
        self.batch_size = batch_size
        
        # Cache for nodes and edges to prevent duplicates
        self.node_cache = set()
        self.edge_cache = set()
        
        # Enhanced rate limiting and retry settings
        self.last_request_time = 0
        self.min_request_interval = 3.0  # Increased to 3 seconds between requests
        self.retry_count = 0
        self.max_retries = 5  # Increased max retries
        self.base_retry_delay = 5  # Base delay for exponential backoff
        self.max_retry_delay = 60  # Maximum retry delay
        
        # Progress tracking
        self.current_chunk = 0
        self.total_chunks = 0
        
        # Add cache file paths
        self.cache_dir = Path("processed_data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.processed_chunks_file = self.cache_dir / "processed_chunks.json"
        self.progress_file = self.cache_dir / "extraction_progress.json"
        self.processed_chunks = self._load_processed_chunks()
        self.progress = self._load_progress()

    def _load_processed_chunks(self) -> Dict[str, Any]:
        """Load previously processed chunks from cache."""
        if self.processed_chunks_file.exists():
            with open(self.processed_chunks_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_processed_chunks(self):
        """Save processed chunks to cache."""
        with open(self.processed_chunks_file, 'w') as f:
            json.dump(self.processed_chunks, f, indent=2)

    def _load_progress(self) -> Dict[str, Any]:
        """Load progress from previous run."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {"last_processed_chunk": 0, "failed_chunks": []}

    def _save_progress(self):
        """Save current progress."""
        with open(self.progress_file, 'w') as f:
            json.dump({
                "last_processed_chunk": self.current_chunk,
                "total_chunks": self.total_chunks,
                "failed_chunks": self.progress.get("failed_chunks", []),
                "timestamp": str(datetime.now())
            }, f, indent=2)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent entity names."""
        if not text:
            return ""
        
        # Convert to uppercase and remove extra whitespace
        text = ' '.join(text.upper().split())
        
        # Remove line breaks and extra spaces
        text = re.sub(r'\s*\n\s*', ' ', text)
        
        # Standardize message names
        text = re.sub(r'(?i)(AUTHENTICATION|Registration|Identity|Security Mode|PDU Session|Deregistration)\s*[\n\r\s]+\s*(Request|Response|Accept|Reject|Command|Complete)', 
                     r'\1 \2', text)
        
        # Standardize arrow notation
        text = re.sub(r'\s*-+>\s*', ' -> ', text)
        
        return text

    def _normalize_entity_name(self, name: str) -> str:
        """Enhanced entity name normalization."""
        if not name:
            return ""
            
        # Common network element mappings
        common_elements = {
            "USER EQUIPMENT": "UE",
            "ACCESS AND MOBILITY MANAGEMENT FUNCTION": "AMF",
            "SESSION MANAGEMENT FUNCTION": "SMF",
            "USER PLANE FUNCTION": "UPF",
            "NEXT GENERATION NODE B": "GNB",
            "AUTHENTICATION SERVER FUNCTION": "AUSF",
            "UNIFIED DATA MANAGEMENT": "UDM",
            "POLICY CONTROL FUNCTION": "PCF",
            "NETWORK REPOSITORY FUNCTION": "NRF"
        }
        
        # Normalize message names
        message_types = ["REQUEST", "RESPONSE", "ACCEPT", "REJECT", "COMMAND", "COMPLETE"]
        
        # First normalize the text
        name = self._normalize_text(name)
        
        # Check if it's a full name that should be mapped to abbreviation
        if name in common_elements:
            return common_elements[name]
        
        # Handle message names
        for msg_type in message_types:
            if msg_type in name:
                parts = name.split()
                if len(parts) > 1:
                    return ' '.join(parts)
        
        # Special handling for timers (preserve format T####)
        if name.startswith('T') and len(name) >= 4 and name[1:5].isdigit():
            return name[:5]
            
        return name

    def _get_chunk_hash(self, chunk: Dict[str, str]) -> str:
        """Generate a hash for a chunk to track changes."""
        content = json.dumps(chunk, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def _wait_for_rate_limit(self):
        """Enhanced rate limiting with adaptive delays and jitter."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        # Base delay with exponential backoff based on retry count
        base_delay = self.min_request_interval * (1.5 ** self.retry_count)
        
        if time_since_last_request < base_delay:
            # Calculate sleep time and add random jitter
            sleep_time = base_delay - time_since_last_request
            jitter = random.uniform(0, min(1.0, sleep_time * 0.1))  # 10% jitter
            total_sleep = sleep_time + jitter
            
            # Cap the maximum sleep time
            total_sleep = min(total_sleep, self.max_retry_delay)
            
            time.sleep(total_sleep)
        
        self.last_request_time = time.time()

    def _handle_api_error(self, e: Exception, chunk_id: int = None) -> bool:
        """Enhanced error handling with better error classification and recovery."""
        error_str = str(e).lower()
        
        # Classify error type
        is_quota_error = "429" in error_str or "quota" in error_str
        is_timeout_error = "timeout" in error_str or "deadline" in error_str
        is_connection_error = "connection" in error_str or "network" in error_str
        
        if any([is_quota_error, is_timeout_error, is_connection_error]):
            if self.retry_count < self.max_retries:
                self.retry_count += 1
                
                # Calculate delay with exponential backoff and error-specific base delays
                if is_quota_error:
                    base_delay = self.base_retry_delay * 2
                elif is_timeout_error:
                    base_delay = self.base_retry_delay * 1.5
                else:  # connection error
                    base_delay = self.base_retry_delay
                
                delay = min(base_delay * (2 ** (self.retry_count - 1)), self.max_retry_delay)
                
                # Add some randomization to prevent thundering herd
                delay *= random.uniform(0.8, 1.2)
                
                error_type = "Quota" if is_quota_error else "Timeout" if is_timeout_error else "Connection"
                print(f"\n{error_type} error encountered. Waiting {delay:.1f} seconds before retry {self.retry_count}/{self.max_retries}")
                
                # Save progress before waiting
                if chunk_id is not None:
                    if "failed_chunks" not in self.progress:
                        self.progress["failed_chunks"] = []
                    if chunk_id not in self.progress["failed_chunks"]:
                        self.progress["failed_chunks"].append(chunk_id)
                    self._save_progress()
                
                time.sleep(delay)
                return True
            else:
                print(f"\nMax retries ({self.max_retries}) reached. Saving progress and continuing with next chunk...")
                self.retry_count = 0
                return False
                
        # For other types of errors, log them but don't retry
        print(f"\nUnexpected error: {str(e)}")
        return False

    def _create_node_extraction_prompt(self, text: str) -> str:
        """Create optimized prompt for node extraction."""
        return f"""Extract network elements, states, and events as a JSON array. Each item must have this structure:
{{
    "node": true,
    "name": "node name",
    "type": "NetworkElement|State|Event",
    "description": "brief description"
}}

Example output:
[
    {{
        "node": true,
        "name": "UE",
        "type": "NetworkElement",
        "description": "User Equipment"
    }}
]

Text to analyze:
{text}

Return ONLY a valid JSON array."""

    def _create_edge_extraction_prompt(self, text: str) -> str:
        """Create optimized prompt for edge extraction."""
        return f"""Extract relationships as a JSON array. Each item must have this structure:
{{
    "edge": true,
    "source": "source node name",
    "target": "target node name",
    "type": "SENDS|TRANSITIONS_TO|TRIGGERS",
    "description": "brief description"
}}

Example output:
[
    {{
        "edge": true,
        "source": "UE",
        "target": "AMF",
        "type": "SENDS",
        "description": "Registration request"
    }}
]

Text to analyze:
{text}

Return ONLY a valid JSON array."""

    def _extract_nodes(self, text: str) -> List[Node]:
        """Extract nodes using JSON format with enhanced debugging."""
        nodes = []
        retry = True
        attempts = 0
        max_attempts = 3
        
        while retry and attempts < max_attempts:
            try:
                self._wait_for_rate_limit()
                
                # Debug: Print input text
                print("\n=== Input Text Preview ===")
                print(f"Length: {len(text)} characters")
                print("First 200 chars:")
                print(text[:200])
                print("=== End Input Text Preview ===\n")
                
                # Get model response
                print("Sending extraction prompt to model...")
                response = self.model.generate_content(
                    self._create_node_extraction_prompt(text),
                    generation_config=self.generation_config
                )
                
                # Debug: Print raw response
                print("\n=== Raw Model Response ===")
                print(response.text)
                print("=== End Raw Model Response ===\n")
                
                try:
                    # Parse JSON response
                    node_data = json.loads(response.text)
                    print(f"\nParsed {len(node_data)} nodes from JSON")
                    
                    valid_node_count = 0
                    for i, node_json in enumerate(node_data, 1):
                        try:
                            # Validate required fields
                            if not all(k in node_json for k in ['node', 'name', 'type', 'description']):
                                print(f"Skipping node {i} - missing required fields")
                                continue
                            
                            if not node_json['node']:
                                print(f"Skipping node {i} - node field is not true")
                                continue
                            
                            # Validate node type
                            if node_json['type'] not in ['NetworkElement', 'State', 'Event']:
                                print(f"Skipping node {i} - invalid type: {node_json['type']}")
                                continue
                            
                            # Create node dictionary
                            node = {
                                'id': hashlib.md5(node_json['name'].encode()).hexdigest()[:8],
                                'type': node_json['type'],
                                'name': node_json['name'],
                                'properties': {
                                    'description': node_json['description']
                                }
                            }
                            
                            # Create Node object
                            nodes.append(Node(**node))
                            valid_node_count += 1
                            print(f"Successfully created node: {node['type']} - {node['name']}")
                            
                        except Exception as e:
                            print(f"Error processing node {i}: {str(e)}")
                            continue
                    
                    print(f"\nExtracted {valid_node_count} valid nodes from {len(node_data)} JSON objects")
                    retry = False
                    
                except json.JSONDecodeError as je:
                    print(f"\nError decoding JSON response: {str(je)}")
                    retry = self._handle_api_error(je)
                    if not retry:
                        break
                    attempts += 1
                    time.sleep(5)
                    
            except Exception as e:
                print(f"\nError in node extraction: {str(e)}")
                retry = self._handle_api_error(e)
                if not retry:
                    break
                attempts += 1
                time.sleep(5)
        
        # Final summary
        print("\n=== Node Extraction Summary ===")
        print(f"Total nodes extracted: {len(nodes)}")
        if nodes:
            print("\nExtracted nodes:")
            for node in nodes:
                print(f"- {node.type}: {node.name}")
        else:
            print("No nodes were extracted!")
        print("=== End Summary ===\n")
        
        return nodes

    def _extract_edges(self, text: str, nodes: List[Node]) -> List[Edge]:
        """Extract edges using JSON format with enhanced debugging."""
        edges = []
        retry = True
        attempts = 0
        max_attempts = 3
        
        # Create node lookup maps
        node_maps = self._create_node_maps(nodes)
        
        # Debug: Print available nodes
        print("\n=== Available Nodes ===")
        print(f"Total nodes: {len(nodes)}")
        for node in nodes:
            print(f"- {node.type}: {node.name}")
        print("=== End Available Nodes ===\n")
        
        while retry and attempts < max_attempts:
            try:
                self._wait_for_rate_limit()
                
                # Debug: Print input text
                print("\n=== Input Text Preview ===")
                print(f"Length: {len(text)} characters")
                print("First 200 chars:")
                print(text[:200])
                print("=== End Input Text Preview ===\n")
                
                # Get model response
                print("Sending edge extraction prompt to model...")
                response = self.model.generate_content(
                    self._create_edge_extraction_prompt(text),
                    generation_config=self.generation_config
                )
                
                # Debug: Print raw response
                print("\n=== Raw Model Response ===")
                print(response.text)
                print("=== End Raw Model Response ===\n")
                
                try:
                    # Parse JSON response
                    edge_data = json.loads(response.text)
                    print(f"\nParsed {len(edge_data)} edges from JSON")
                    
                    valid_edge_count = 0
                    for i, edge_json in enumerate(edge_data, 1):
                        try:
                            # Validate required fields
                            if not all(k in edge_json for k in ['edge', 'source', 'target', 'type', 'description']):
                                print(f"Skipping edge {i} - missing required fields")
                                continue
                            
                            if not edge_json['edge']:
                                print(f"Skipping edge {i} - edge field is not true")
                                continue
                            
                            # Match source and target nodes
                            source_match = self._match_node(edge_json['source'], node_maps)
                            if not source_match:
                                print(f"Skipping edge {i} - source node not found: {edge_json['source']}")
                                continue
                            
                            target_match = self._match_node(edge_json['target'], node_maps)
                            if not target_match:
                                print(f"Skipping edge {i} - target node not found: {edge_json['target']}")
                                continue
                            
                            # Validate edge type
                            edge_type = edge_json['type'].upper()
                            if edge_type not in ['SENDS', 'TRANSITIONS_TO', 'TRIGGERS']:
                                print(f"Skipping edge {i} - invalid type: {edge_type}")
                                continue
                            
                            # Create edge dictionary
                            edge = {
                                'source': source_match,
                                'target': target_match,
                                'type': edge_type,
                                'properties': {
                                    'description': edge_json['description']
                                }
                            }
                            
                            # Create Edge object
                            edges.append(Edge(**edge))
                            valid_edge_count += 1
                            print(f"Successfully created edge: {edge_json['source']} -{edge_type}-> {edge_json['target']}")
                            
                        except Exception as e:
                            print(f"Error processing edge {i}: {str(e)}")
                            continue
                    
                    print(f"\nExtracted {valid_edge_count} valid edges from {len(edge_data)} JSON objects")
                    retry = False
                    
                except json.JSONDecodeError as je:
                    print(f"\nError decoding JSON response: {str(je)}")
                    retry = self._handle_api_error(je)
                    if not retry:
                        break
                    attempts += 1
                    time.sleep(5)
                    
            except Exception as e:
                print(f"\nError in edge extraction: {str(e)}")
                retry = self._handle_api_error(e)
                if not retry:
                    break
                attempts += 1
                time.sleep(5)
        
        # Final summary
        print("\n=== Edge Extraction Summary ===")
        print(f"Total edges extracted: {len(edges)}")
        if edges:
            print("\nExtracted edges:")
            for edge in edges:
                print(f"- {edge.source} -{edge.type}-> {edge.target}")
        else:
            print("No edges were extracted!")
        print("=== End Summary ===\n")
        
        return edges

    def _create_node_maps(self, nodes: List[Node]) -> Dict[str, Dict[str, str]]:
        """Create lookup maps for node matching."""
        maps = {
            "by_id": {},
            "by_name": {},
            "by_normalized": {}
        }
        
        for node in nodes:
            maps["by_id"][node.id] = node.id
            maps["by_name"][node.name] = node.id
            norm_name = self._normalize_entity_name(node.name)
            maps["by_normalized"][norm_name] = node.id
            maps["by_normalized"][norm_name.upper()] = node.id
            maps["by_normalized"][norm_name.replace(" ", "")] = node.id
            
        return maps

    def _match_node(self, node_ref: str, node_maps: Dict[str, Dict[str, str]]) -> str:
        """Match node reference to known node ID."""
        if not node_ref:
            return None
            
        # Try direct ID match
        if node_ref in node_maps["by_id"]:
            return node_maps["by_id"][node_ref]
            
        # Try name match
        if node_ref in node_maps["by_name"]:
            return node_maps["by_name"][node_ref]
            
        # Try normalized matches
        norm_ref = self._normalize_entity_name(node_ref)
        if norm_ref in node_maps["by_normalized"]:
            return node_maps["by_normalized"][norm_ref]
            
        return None

    def _process_chunk_batch(self, chunks: List[Dict[str, str]]) -> Tuple[List[Node], List[Edge]]:
        """Process a batch of chunks with optimized performance."""
        try:
            # Generate batch ID for Neo4j tracking
            batch_id = f"batch_{int(time.time())}_{random.randint(1000, 9999)}"
            
            # Combine chunks with clear separators
            combined_text = "\n\n=== SECTION BREAK ===\n\n".join([chunk['content'] for chunk in chunks])
            
            # Clean and normalize text
            cleaned_text = self._normalize_text(combined_text)
            
            # Split into optimal-sized chunks (4000 chars with 500 char overlap)
            chunk_size = 4000
            overlap = 500
            text_chunks = []
            
            for i in range(0, len(cleaned_text), chunk_size - overlap):
                chunk = cleaned_text[i:i + chunk_size]
                if chunk:
                    text_chunks.append(chunk)
            
            print(f"\nProcessing {len(text_chunks)} optimized chunks...")
            
            # Process chunks in parallel
            all_nodes = []
            all_edges = []
            node_cache = set()
            
            # First extract all nodes
            for chunk in text_chunks:
                nodes = self._extract_nodes(chunk)
                for node in nodes:
                    node_key = f"{node.type}:{node.name}"
                    if node_key not in node_cache:
                        node_cache.add(node_key)
                        all_nodes.append(node)
            
            print(f"\nExtracted {len(all_nodes)} unique nodes")
            
            # Then extract edges using all known nodes
            if all_nodes:
                for chunk in text_chunks:
                    edges = self._extract_edges(chunk, all_nodes)
                    all_edges.extend(edges)
                
                print(f"Extracted {len(all_edges)} edges")
                
                # Store in Neo4j
                try:
                    from store_data_neo4j import store_batch_in_neo4j
                    
                    # Convert to dicts for storage
                    node_dicts = [
                        {
                            "id": node.id,
                            "type": node.type,
                            "name": node.name,
                            "properties": node.properties
                        } 
                        for node in all_nodes
                    ]
                    
                    edge_dicts = [
                        {
                            "source": edge.source,
                            "target": edge.target,
                            "type": edge.type,
                            "properties": edge.properties
                        }
                        for edge in all_edges
                    ]
                    
                    # Store in single batch
                    store_batch_in_neo4j(node_dicts, edge_dicts, batch_id)
                    
                except Exception as e:
                    print(f"Warning: Error storing in Neo4j: {str(e)}")
            
            return all_nodes, all_edges
            
        except Exception as e:
            print(f"Error in batch processing: {str(e)}")
            return [], []

    def process_pdf(self, pdf_path: str, output_file: str = None, max_chunks: int = 500, load_to_neo4j: bool = True):
        """Enhanced PDF processing with better quota handling and progress tracking."""
        try:
            pdf_path = str(Path(pdf_path).resolve())
            
            # Create processed_data directory if it doesn't exist
            Path("processed_data").mkdir(parents=True, exist_ok=True)
            
            # Step 1: Preprocess the PDF
            print(f"Processing PDF: {pdf_path}")
            documents = read_pdfs_from_directory(
                directory=str(Path(pdf_path).parent),
                pattern=Path(pdf_path).name
            )
            
            if not documents:
                print("No content found in PDF")
                return None
            
            # Save preprocessed chunks to JSON for inspection
            preprocessed_output = {
                "total_chunks": len(documents),
                "chunks": documents,
                "metadata": {
                    "source_file": pdf_path,
                    "timestamp": str(datetime.now())
                }
            }
            
            preprocessed_file = "processed_data/preprocessed_chunks.json"
            with open(preprocessed_file, 'w', encoding='utf-8') as f:
                json.dump(preprocessed_output, f, indent=2)
            print(f"\nPreprocessed chunks saved to {preprocessed_file}")
            
            # Limit chunks and track which ones need processing
            documents = documents[:max_chunks] if max_chunks else documents
            self.total_chunks = len(documents)
            
            print(f"\nTotal chunks to process: {self.total_chunks}")
            
            # Start from last processed chunk
            start_chunk = self.progress.get("last_processed_chunk", 0)
            chunks_to_process = []
            
            for i, chunk in enumerate(documents[start_chunk:], start=start_chunk):
                chunk_hash = self._get_chunk_hash(chunk)
                if chunk_hash not in self.processed_chunks:
                    chunks_to_process.append((i, chunk))
            
            if not chunks_to_process:
                print("No new content to process")
                if existing_graph:
                    print("Using existing graph data")
                    return existing_graph
                return None
            
            print(f"Processing {len(chunks_to_process)} new chunks...")
            
            # Step 2: Process chunks with enhanced error handling
            all_nodes = []
            all_edges = []
            failed_chunks = []
            
            # Create smaller batches for better quota management
            batch_size = self.batch_size
            batches = []
            for i in range(0, len(chunks_to_process), batch_size):
                batch = chunks_to_process[i:i + batch_size]
                batches.append(batch)
            
            print(f"Processing chunks in {len(batches)} batches...")
            
            # Create a file to log extraction results
            extraction_log = "processed_data/extraction_log.json"
            extraction_results = {
                "batches": [],
                "failed_chunks": [],
                "metadata": {
                    "source_file": pdf_path,
                    "start_time": str(datetime.now())
                }
            }
            
            for batch_idx, batch in enumerate(batches):
                try:
                    print(f"\nProcessing batch {batch_idx + 1}/{len(batches)}")
                    batch_chunks = [chunk for _, chunk in batch]
                    nodes, edges = self._process_chunk_batch(batch_chunks)
                    
                    # Log the results for this batch
                    batch_result = {
                        "batch_index": batch_idx,
                        "chunk_indices": [idx for idx, _ in batch],
                        "nodes_extracted": len(nodes),
                        "edges_extracted": len(edges),
                        "node_types": {},
                        "edge_types": {}
                    }
                    
                    # Count node types
                    for node in nodes:
                        batch_result["node_types"][node.type] = batch_result["node_types"].get(node.type, 0) + 1
                    
                    # Count edge types
                    for edge in edges:
                        batch_result["edge_types"][edge.type] = batch_result["edge_types"].get(edge.type, 0) + 1
                    
                    extraction_results["batches"].append(batch_result)
                    
                    # Update progress
                    self.current_chunk = batch[-1][0]  # Last chunk in batch
                    self._save_progress()
                    
                    # Deduplicate and store results
                    for node in nodes:
                        node_key = f"{node.type}:{node.name}"
                        if node_key not in self.node_cache:
                            self.node_cache.add(node_key)
                            all_nodes.append(node)
                    
                    for edge in edges:
                        edge_key = f"{edge.source}:{edge.type}:{edge.target}"
                        if edge_key not in self.edge_cache:
                            self.edge_cache.add(edge_key)
                            all_edges.append(edge)
                    
                    # Save extraction results after each batch
                    extraction_results["metadata"]["end_time"] = str(datetime.now())
                    extraction_results["metadata"]["total_nodes"] = len(all_nodes)
                    extraction_results["metadata"]["total_edges"] = len(all_edges)
                    with open(extraction_log, 'w', encoding='utf-8') as f:
                        json.dump(extraction_results, f, indent=2)
                    
                    # Reset retry count after successful batch
                    self.retry_count = 0
                    
                    # Add longer pause between batches
                    time.sleep(self.min_request_interval * 2)
                    
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {str(e)}")
                    failed_chunks.extend([idx for idx, _ in batch])
                    extraction_results["failed_chunks"].extend([idx for idx, _ in batch])
                    
                    # Save progress before potential exit
                    self._save_progress()
                    
                    if not self._handle_api_error(e, self.current_chunk):
                        print("Saving current progress and results...")
                        break
            
            # Save final results
            graph_data = {
                "nodes": [{"id": n.id, "type": n.type, "name": n.name, "properties": n.properties} 
                         for n in all_nodes],
                "edges": [{"source": e.source, "target": e.target, "type": e.type, "properties": e.properties} 
                         for e in all_edges],
                "metadata": {
                    "source_document": pdf_path,
                    "total_chunks": self.total_chunks,
                    "processed_chunks": self.current_chunk + 1,
                    "failed_chunks": failed_chunks,
                    "extraction_timestamp": str(datetime.now())
                }
            }
            
            # Save detailed statistics
            stats_file = "processed_data/extraction_stats.json"
            statistics = {
                "node_type_distribution": {},
                "edge_type_distribution": {},
                "total_nodes": len(all_nodes),
                "total_edges": len(all_edges),
                "failed_chunks": failed_chunks,
                "processing_time": str(datetime.now() - datetime.fromisoformat(extraction_results["metadata"]["start_time"].replace("Z", "+00:00")))
            }
            
            # Calculate distributions
            for node in all_nodes:
                statistics["node_type_distribution"][node.type] = statistics["node_type_distribution"].get(node.type, 0) + 1
            
            for edge in all_edges:
                statistics["edge_type_distribution"][edge.type] = statistics["edge_type_distribution"].get(edge.type, 0) + 1
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(statistics, f, indent=2)
            
            # Save to file if specified
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open('w', encoding='utf-8') as f:
                    json.dump(graph_data, f, indent=2)
                print(f"\nGraph data saved to {output_file}")
            
            print(f"\nExtracted {len(all_nodes)} nodes and {len(all_edges)} edges from {self.total_chunks} chunks")
            print(f"Failed chunks: {len(failed_chunks)}")
            print(f"\nDetailed statistics saved to {stats_file}")
            print(f"Extraction log saved to {extraction_log}")
            
            # Load data into Neo4j if requested
            if load_to_neo4j:
                try:
                    from store_data_neo4j import load_graph_data
                    print("\nLoading data into Neo4j...")
                    if output_file:
                        load_graph_data(output_file)
                    else:
                        # Create temporary file if no output file specified
                        temp_file = "processed_data/temp_graph.json"
                        with open(temp_file, 'w', encoding='utf-8') as f:
                            json.dump(graph_data, f, indent=2)
                        load_graph_data(temp_file)
                        # Clean up temp file
                        os.remove(temp_file)
                    print("✓ Data successfully loaded into Neo4j")
                except Exception as e:
                    print(f"Error loading data into Neo4j: {str(e)}")
            
            # After processing, save cache
            self._save_processed_chunks()
            
            return graph_data
            
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return None

    def visualize_graph(self, graph_data: Dict[str, Any]):
        """Create an interactive visualization of the extracted graph."""
        # Use plt.ion() for interactive mode
        plt.ion()
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes with attributes
        node_colors = {
            'network_element': '#ff7f0e',  # Orange
            'message': '#1f77b4',         # Blue
            'state': '#2ca02c',           # Green
            'timer': '#d62728',           # Red
            'protocol': '#9467bd',        # Purple
            'procedure': '#8c564b'        # Brown
        }
        
        # Add nodes
        for node in graph_data['nodes']:
            G.add_node(node['id'], 
                      node_type=node['type'],
                      name=node['name'],
                      color=node_colors.get(node['type'], '#7f7f7f'))
        
        # Add edges
        for edge in graph_data['edges']:
            G.add_edge(edge['source'], edge['target'], 
                      edge_type=edge['type'],
                      **edge['properties'])
        
        # Create the visualization
        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        for node_type in node_colors:
            node_list = [node for node in G.nodes() 
                        if G.nodes[node]['node_type'] == node_type]
            nx.draw_networkx_nodes(G, pos, 
                                 nodelist=node_list,
                                 node_color=node_colors[node_type],
                                 node_size=1000,
                                 alpha=0.6,
                                 label=node_type)
        
        # Draw edges with different colors based on type
        edge_colors = defaultdict(lambda: '#666666')
        edge_colors.update({
            'sends': '#1f77b4',
            'transitions_to': '#2ca02c',
            'triggers': '#d62728',
            'requires': '#9467bd',
            'establishes': '#8c564b'
        })
        
        for edge_type in set(nx.get_edge_attributes(G, 'edge_type').values()):
            edge_list = [(u, v) for (u, v, d) in G.edges(data=True) 
                        if d['edge_type'] == edge_type]
            nx.draw_networkx_edges(G, pos,
                                 edgelist=edge_list,
                                 edge_color=edge_colors[edge_type],
                                 alpha=0.5,
                                 label=edge_type)
        
        # Add labels
        labels = {node: G.nodes[node]['name'] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title("3GPP Specification Graph", fontsize=16, pad=20)
        plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.axis('off')
        
        # Enable interaction
        plt.draw()
        plt.pause(0.001)  # Small pause to allow the window to update
        
        # Keep the plot window open
        input("Press Enter to close the visualization...")
        plt.close()

def main():
    # Example usage
    api_key = os.getenv("GOOGLE_API_KEY")
    extractor = ThreeGPPGraphExtractor(api_key, batch_size=5)  # Smaller batch size for reliability
    
    # Process the PDF file
    pdf_file = "data/TS 24.501.pdf"
    
    if not Path(pdf_file).exists():
        print(f"Error: PDF file not found - {pdf_file}")
        return
    
    # Process the entire document
    results = extractor.process_pdf(
        pdf_path=pdf_file,
        output_file="processed_data/3gpp_graph.json"
        # No max_chunks parameter, so it will process all chunks
    )
    
    if results:
        # Print statistics
        node_types = {}
        edge_types = {}
        
        for node in results["nodes"]:
            node_types[node["type"]] = node_types.get(node["type"], 0) + 1
        
        for edge in results["edges"]:
            edge_types[edge["type"]] = edge_types.get(edge["type"], 0) + 1
        
        print("\nExtraction Summary:")
        print("-" * 50)
        print(f"Total nodes extracted: {len(results['nodes'])}")
        print(f"Total edges extracted: {len(results['edges'])}")
        
        print("\nNode types distribution:")
        for ntype, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
            print(f"- {ntype}: {count}")
        
        print("\nEdge types distribution:")
        for etype, count in sorted(edge_types.items(), key=lambda x: x[1], reverse=True):
            print(f"- {etype}: {count}")
        
        print("\nSaving visualization...")
        # Create interactive visualization
        extractor.visualize_graph(results)
if __name__ == "__main__":
    main() 
