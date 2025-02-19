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
        """Enhanced rate limiting with adaptive delays."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            # Add small random jitter to prevent synchronized requests
            sleep_time += random.uniform(0, 1)
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def _handle_api_error(self, e: Exception, chunk_id: int = None) -> bool:
        """Enhanced error handling with exponential backoff."""
        if "429" in str(e) or "quota" in str(e).lower():
            if self.retry_count < self.max_retries:
                self.retry_count += 1
                # Calculate delay with exponential backoff
                delay = min(self.base_retry_delay * (2 ** (self.retry_count - 1)), self.max_retry_delay)
                print(f"\nQuota error encountered. Waiting {delay} seconds before retry {self.retry_count}/{self.max_retries}")
                
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
                print("\nMax retries reached. Saving progress and continuing with next chunk...")
                self.retry_count = 0
                return False
        return False

    def _create_node_extraction_prompt(self, text: str) -> str:
        """Create prompt for node extraction."""
        return f"""
        Analyze this 3GPP specification text and extract nodes for all procedures.
        Focus on identifying these types of nodes:

        1. State Nodes:
           - Initial states (e.g., IDLE, DEREGISTERED)
           - Intermediate states (e.g., CONNECTING, AUTHENTICATING)
           - Final states (e.g., CONNECTED, REGISTERED)
           Properties to include:
           - state_type: "initial", "intermediate", or "final"
           - entry_conditions: conditions required to enter this state
           - exit_conditions: conditions that trigger exit from this state
           - description: detailed description of the state
           - metadata: {{
               "timeout": timeout value if applicable,
               "retry_count": max retry count if applicable,
               "security_context": security context if applicable,
               "procedure": procedure this state belongs to
           }}

        2. Event Nodes:
           - Message events (e.g., REQUEST_RECEIVED, RESPONSE_SENT)
           - Timer events (e.g., T3510_EXPIRED, GUARD_TIMER_EXPIRED)
           - Internal events (e.g., SECURITY_CONTEXT_CREATED)
           Properties to include:
           - event_type: "message", "timer", or "internal"
           - source_entity: entity that generates the event
           - target_entity: entity that receives the event
           - parameters: list of parameters associated with the event
           - metadata: {{
               "protocol": protocol used (e.g., "NAS", "RRC", "NGAP"),
               "message_type": message type if applicable,
               "timer_value": timer duration if applicable,
               "retry_count": max retry count if applicable,
               "procedure": procedure this event belongs to
           }}

        3. Network Element Nodes:
           - Core Network elements (e.g., AMF, SMF, UPF)
           - Access Network elements (e.g., gNB, ng-eNB)
           - User Equipment (UE)
           Properties to include:
           - element_type: "core_network", "access_network", or "user_equipment"
           - role: primary function in the network
           - interfaces: list of supported interfaces
           - metadata: {{
               "network_type": "5G", "4G", etc.,
               "supported_procedures": list of procedures this element participates in
           }}

        4. Conditional Nodes:
           - Decision points in procedures
           Properties to include:
           - condition_type: "validation", "capability_check", "timer_check", etc.
           - true_path: action/state when condition is true
           - false_path: action/state when condition is false
           - parameters: list of parameters involved in the condition
           - metadata: {{
               "procedure": procedure this condition belongs to,
               "retry_allowed": whether retry is allowed on false path,
               "error_handling": how errors are handled
           }}

        5. Parameter Nodes:
           - Message parameters (e.g., IMSI, GUTI, TAI)
           - Configuration parameters
           Properties to include:
           - parameter_type: "identifier", "capability", "configuration", etc.
           - format: data format or structure
           - mandatory: whether parameter is mandatory
           - metadata: {{
               "procedures": list of procedures using this parameter,
               "validation_rules": rules for parameter validation
           }}

        Text to analyze:
        {text}

        Return ONLY a JSON object with this exact structure:
        {{
            "nodes": [
                {{
                    "id": "node_name",
                    "type": "State|Event|NetworkElement|Conditional|Parameter",
                    "properties": {{
                        // Include appropriate properties based on node type as described above
                        "state_type": "state_type_value",  // For State nodes
                        "event_type": "event_type_value",  // For Event nodes
                        "element_type": "element_type_value",  // For NetworkElement nodes
                        "condition_type": "condition_type_value",  // For Conditional nodes
                        "parameter_type": "parameter_type_value",  // For Parameter nodes
                        // Common properties
                        "description": "detailed description",
                        "metadata": {{
                            // Include appropriate metadata based on node type
                        }}
                    }}
                }}
            ]
        }}
        """

    def _create_edge_extraction_prompt(self, text: str) -> str:
        """Create prompt for edge extraction."""
        return f"""
        Analyze this 3GPP specification text and extract edges (relationships) between nodes.
        Focus on these types of relationships:

        1. State Transitions:
           - Between states in the same procedure
           - Cross-procedure transitions
           Properties to include:
           - trigger_event: event causing the transition
           - conditions: list of conditions that must be met
           - parameters: list of parameters involved
           - metadata: {{
               "procedure": procedure this transition belongs to,
               "protocol": protocol used,
               "timer_value": timer duration if applicable,
               "security_context": security context if applicable
           }}

        2. Message Flows:
           - Between network elements
           - Protocol-specific interactions
           Properties to include:
           - message_type: type of message
           - direction: "uplink" or "downlink"
           - parameters: list of parameters in the message
           - metadata: {{
               "procedure": procedure this message belongs to,
               "protocol": protocol used,
               "security_required": whether security is required,
               "retry_behavior": retry behavior if message fails
           }}

        3. Timer Relationships:
           - Timer start/stop events
           - Timer expiry actions
           Properties to include:
           - timer_action: "start", "stop", or "expire"
           - target_state: state to transition to on expiry
           - metadata: {{
               "procedure": procedure this timer belongs to,
               "duration": timer duration,
               "retry_count": number of retries allowed
           }}

        4. Conditional Flows:
           - Decision paths
           - Error handling paths
           Properties to include:
           - condition: the condition being evaluated
           - true_path: next step if condition is true
           - false_path: next step if condition is false
           - metadata: {{
               "procedure": procedure this flow belongs to,
               "error_handling": how errors are handled,
               "retry_allowed": whether retry is allowed
           }}

        5. Parameter Dependencies:
           - Parameter validations
           - Parameter requirements
           Properties to include:
           - dependency_type: "requires", "validates", or "configures"
           - validation_rules: rules for parameter validation
           - metadata: {{
               "procedure": procedure this dependency belongs to,
               "mandatory": whether dependency is mandatory
           }}

        6. Procedure Links:
           - Between different procedures
           - Sub-procedure relationships
           Properties to include:
           - link_type: "triggers", "depends_on", or "includes"
           - conditions: conditions for the link
           - metadata: {{
               "network_type": network type (4G/5G),
               "priority": priority of the link,
               "fallback": fallback procedure if available
           }}

        Text to analyze:
        {text}

        Return ONLY a JSON object with this exact structure:
        {{
            "edges": [
                {{
                    "source": "source_node_name",
                    "target": "target_node_name",
                    "type": "transitions_to|sends|triggers|requires|validates|includes",
                    "properties": {{
                        // Include appropriate properties based on edge type
                        "trigger_event": "event_name",  // For state transitions
                        "message_type": "message_type",  // For message flows
                        "timer_action": "action_type",  // For timer relationships
                        "condition": "condition_expr",  // For conditional flows
                        "dependency_type": "dep_type",  // For parameter dependencies
                        "link_type": "link_type",  // For procedure links
                        // Common properties
                        "parameters": ["param1", "param2"],
                        "conditions": ["condition1", "condition2"],
                        "metadata": {{
                            "procedure": "procedure_name",
                            "protocol": "protocol_name",
                            // Additional metadata based on edge type
                        }}
                    }}
                }}
            ]
        }}
        """

    def extract_nodes_and_edges(self, text: str) -> Tuple[List[Node], List[Edge]]:
        """Extract both nodes and edges from the text."""
        nodes = []
        edges = []
        
        try:
            # Clean and normalize the text first
            cleaned_text = self._normalize_text(text)
            
            # Extract nodes first
            retry = True
            node_extraction_attempts = 0
            max_attempts = 3
            
            while retry and node_extraction_attempts < max_attempts:
                try:
                    self._wait_for_rate_limit()
                    node_response = self.model.generate_content(
                        self._create_node_extraction_prompt(cleaned_text),
                        generation_config=self.generation_config
                    )
                    
                    # Clean and parse the response
                    node_text = node_response.text
                    
                    # Clean the response text
                    # First remove any markdown code block indicators
                    node_text = re.sub(r'```(?:json)?', '', node_text)
                    node_text = re.sub(r'```', '', node_text)
                    node_text = node_text.strip()
                    
                    # Remove comments
                    node_text = re.sub(r'//.*$', '', node_text, flags=re.MULTILINE)
                    
                    # Remove trailing commas in objects and arrays
                    node_text = re.sub(r',(\s*[}\]])', r'\1', node_text)
                    
                    try:
                        node_data = json.loads(node_text)
                        if isinstance(node_data, dict) and "nodes" in node_data:
                            for node in node_data.get("nodes", []):
                                try:
                                    # Validate node structure
                                    if not isinstance(node, dict):
                                        print(f"Warning: Invalid node format: {node}")
                                        continue
                                    
                                    required_fields = ["id", "type", "name"]
                                    if not all(field in node for field in required_fields):
                                        missing = [f for f in required_fields if f not in node]
                                        print(f"Warning: Node missing required fields {missing}: {node}")
                                        continue
                                    
                                    # Normalize node type
                                    node["type"] = node["type"].strip().title()
                                    if node["type"] not in ["State", "Event", "NetworkElement", "Conditional", "Parameter"]:
                                        print(f"Warning: Invalid node type '{node['type']}', skipping")
                                        continue
                                    
                                    # Ensure properties is a dict
                                    if "properties" not in node or not isinstance(node["properties"], dict):
                                        node["properties"] = {}
                                    
                                    # Create the node
                                    nodes.append(Node(**node))
                                    
                                except Exception as ne:
                                    print(f"Error processing node: {str(ne)}")
                                    continue
                            retry = False
                        else:
                            print("Warning: Invalid node data format")
                            node_extraction_attempts += 1
                            
                    except json.JSONDecodeError as je:
                        print(f"Error parsing node JSON: {str(je)}")
                        print("Response text:")
                        print(node_text[:200] + "..." if len(node_text) > 200 else node_text)
                        node_extraction_attempts += 1
                    
                except Exception as e:
                    retry = self._handle_api_error(e)
                    if not retry:
                        print(f"Error extracting nodes: {str(e)}")
                        break
                    node_extraction_attempts += 1
            
            if node_extraction_attempts >= max_attempts:
                print("Warning: Max node extraction attempts reached")
            
            # Create normalized node name mappings
            valid_nodes = {}
            for node in nodes:
                # Store multiple variations of the name
                norm_name = self._normalize_entity_name(node.name)
                valid_nodes[norm_name] = node.name
                valid_nodes[node.name.upper()] = node.name
                valid_nodes[node.name] = node.name
                valid_nodes[node.name.replace(" ", "")] = node.name
            
            # Extract edges
            retry = True
            edge_extraction_attempts = 0
            edge_set = set()
            
            while retry and edge_extraction_attempts < max_attempts:
                try:
                    self._wait_for_rate_limit()
                    edge_response = self.model.generate_content(
                        self._create_edge_extraction_prompt(cleaned_text),
                        generation_config=self.generation_config
                    )
                    
                    # Clean and parse the response
                    edge_text = edge_response.text
                    
                    # Clean the response text
                    # First remove any markdown code block indicators
                    edge_text = re.sub(r'```(?:json)?', '', edge_text)
                    edge_text = re.sub(r'```', '', edge_text)
                    edge_text = edge_text.strip()
                    
                    # Remove comments
                    edge_text = re.sub(r'//.*$', '', edge_text, flags=re.MULTILINE)
                    
                    # Remove trailing commas in objects and arrays
                    edge_text = re.sub(r',(\s*[}\]])', r'\1', edge_text)
                    
                    try:
                        edge_data = json.loads(edge_text)
                        if isinstance(edge_data, dict) and "edges" in edge_data:
                            for edge in edge_data.get("edges", []):
                                try:
                                    # Validate edge structure
                                    if not isinstance(edge, dict):
                                        print(f"Warning: Invalid edge format: {edge}")
                                        continue
                                    
                                    required_fields = ["source", "target", "type"]
                                    if not all(field in edge for field in required_fields):
                                        missing = [f for f in required_fields if f not in edge]
                                        print(f"Warning: Edge missing required fields {missing}: {edge}")
                                        continue
                                    
                                    source = self._normalize_entity_name(edge["source"])
                                    target = self._normalize_entity_name(edge["target"])
                                    
                                    # Try different normalizations to find a match
                                    source_variations = [
                                        source,
                                        source.upper(),
                                        source.replace(" ", ""),
                                        source.strip()
                                    ]
                                    target_variations = [
                                        target,
                                        target.upper(),
                                        target.replace(" ", ""),
                                        target.strip()
                                    ]
                                    
                                    source_match = next((valid_nodes[var] for var in source_variations 
                                                      if var in valid_nodes), None)
                                    target_match = next((valid_nodes[var] for var in target_variations 
                                                      if var in valid_nodes), None)
                                    
                                    if source_match and target_match:
                                        # Normalize edge type
                                        edge["type"] = edge["type"].strip().upper()
                                        
                                        # Ensure properties is a dict
                                        if "properties" not in edge or not isinstance(edge["properties"], dict):
                                            edge["properties"] = {}
                                        
                                        # Create edge key for deduplication
                                        edge_key = f"{source_match}:{edge['type']}:{target_match}"
                                        if edge_key not in edge_set:
                                            edge_set.add(edge_key)
                                            edge["source"] = source_match
                                            edge["target"] = target_match
                                            edges.append(Edge(**edge))
                                    else:
                                        if not source_match:
                                            print(f"Warning: Source node '{source}' not found in valid nodes")
                                        if not target_match:
                                            print(f"Warning: Target node '{target}' not found in valid nodes")
                                except Exception as ee:
                                    print(f"Error processing edge: {str(ee)}")
                                    continue
                            retry = False
                        else:
                            print("Warning: Invalid edge data format")
                            edge_extraction_attempts += 1
                            
                    except json.JSONDecodeError as je:
                        print(f"Error parsing edge JSON: {str(je)}")
                        print("Response text:")
                        print(edge_text[:200] + "..." if len(edge_text) > 200 else edge_text)
                        edge_extraction_attempts += 1
                    
                except Exception as e:
                    retry = self._handle_api_error(e)
                    if not retry:
                        print(f"Error extracting edges: {str(e)}")
                        break
                    edge_extraction_attempts += 1
            
            if edge_extraction_attempts >= max_attempts:
                print("Warning: Max edge extraction attempts reached")
            
            if not edges and nodes:
                print(f"Found {len(nodes)} nodes but no edges were extracted.")
            elif edges:
                print(f"Successfully extracted {len(nodes)} nodes and {len(edges)} edges")
            
            return nodes, edges
            
        except Exception as e:
            print(f"Error in extract_nodes_and_edges: {str(e)}")
            return [], []

    def _process_chunk_batch(self, chunks: List[Dict[str, str]]) -> Tuple[List[Node], List[Edge]]:
        """Process a batch of chunks together."""
        combined_text = "\n\n".join([chunk['content'] for chunk in chunks])
        return self.extract_nodes_and_edges(combined_text)

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
