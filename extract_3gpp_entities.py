import os
import re
import json
from typing import List, Dict, Tuple, Set
from enum import Enum
from dataclasses import dataclass
from pypdf import PdfReader
from neo4j import GraphDatabase
from preprocess_pdfs import read_pdfs_from_directory
from rich.console import Console
from tqdm import tqdm
import hashlib
from dotenv import load_dotenv
from store_data_neo4j import KnowledgeGraph
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define enums and data classes for state management
class StateType(Enum):
    EMM = "EMM"
    MM = "MM"
    GMM = "GMM"
    PMM = "PMM"
    CM = "CM"
    RRC = "RRC"
    FIVE_GMM = "5GMM"

@dataclass
class State:
    name: str
    type: StateType
    description: str
    source_section: str
    conditions: List[str] = None

    def __post_init__(self):
        if self.conditions is None:
            self.conditions = []

@dataclass
class Action:
    name: str
    actor: str
    target: str = None
    outcome: str = None
    parameters: List[str] = None
    prerequisites: List[str] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = []
        if self.prerequisites is None:
            self.prerequisites = []

@dataclass
class ExecutionStep:
    step_number: int
    actor: str
    action: str
    parameters: List[str] = None
    conditions: List[str] = None
    next_steps: List[int] = None
    alternative_steps: Dict[str, int] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = []
        if self.conditions is None:
            self.conditions = []
        if self.next_steps is None:
            self.next_steps = []
        if self.alternative_steps is None:
            self.alternative_steps = {}

# Load environment variables
load_dotenv()

# Get Neo4j credentials from environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

console = Console()
CACHE_FILE = "processed_docs_cache.json"

def get_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of a file"""
    # Normalize path for consistent comparison
    file_path = os.path.normpath(file_path)
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read the file in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def load_cache() -> Dict:
    """Load processed documents from cache"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load cache file: {str(e)}[/yellow]")
    return {}

def save_cache(cache: Dict):
    """Save processed documents to cache"""
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not save cache file: {str(e)}[/yellow]")

class ThreeGPPEntityExtractor:
    def __init__(self):
        # Core regex patterns for 3GPP NAS domain with focus on 5G Registration
        self.patterns = {
            # 5G Registration specific patterns
            "REGISTRATION_PROCEDURE": [
                r'(?:Initial|Mobility|Periodic|Emergency)\s+Registration\s+(?:procedure|Procedure)',
                r'Registration\s+(?:Request|Accept|Complete|Reject)',
                r'(?:UE|AMF)-initiated\s+Registration',
                r'5G(?:S|MM)\s+Registration'
            ],
            
            # 5G States
            "5G_STATE": [
                r'5GMM-(?:REGISTERED|DEREGISTERED|IDLE|CONNECTED)',
                r'5GMM-(?:REGISTERED)\.(?:NORMAL-SERVICE|UPDATE-NEEDED|ATTEMPTING-TO-UPDATE|LIMITED-SERVICE)',
                r'5GMM-(?:DEREGISTERED)\.(?:NORMAL-SERVICE|ATTEMPTING-REGISTRATION|LIMITED-SERVICE|NO-CELL-AVAILABLE)',
                r'RM-(?:REGISTERED|DEREGISTERED)'
            ],
            
            # 5G Network Elements
            "5G_NETWORK_ELEMENT": [
                r'(?:AMF|SMF|UDM|AUSF|SEAF|UPF|gNB|ng-eNB)',
                r'Access and Mobility Management Function',
                r'Session Management Function',
                r'Authentication Server Function',
                r'Security Anchor Function'
            ],
            
            # 5G Messages
            "5G_MESSAGE": [
                r'Registration\s+(?:Request|Accept|Complete|Reject)',
                r'Authentication\s+(?:Request|Response|Result|Reject)',
                r'Security Mode\s+(?:Command|Complete|Reject)',
                r'Identity\s+(?:Request|Response)',
                r'Configuration Update\s+Command'
            ],
            
            # 5G Parameters
            "5G_PARAMETER": [
                r'5G-GUTI',
                r'SUCI',
                r'SUPI',
                r'5G-S-TMSI',
                r'PEI',
                r'(?:Allowed|Rejected)\s+NSSAI',
                r'(?:Request|Configured)\s+NSSAI',
                r'TAI',
                r'5GS\s+(?:registration|update)\s+type',
                r'5GMM\s+capability',
                r'UE\s+security\s+capability'
            ],
            
            # 5G Security
            "5G_SECURITY": [
                r'5G-(?:AKA|EAP)',
                r'SUCI/SUPI\s+de-concealment',
                r'5G\s+security\s+context',
                r'5G\s+NAS\s+security',
                r'KAMF',
                r'5G\s+(?:integrity|ciphering)\s+key'
            ],
            
            # 5G Timers
            "5G_TIMER": [
                r'T3(?:510|511|512|513|515|516|517|520|522|525|540)',
                r'Mobile\s+reachable\s+timer',
                r'Implicit\s+de-registration\s+timer'
            ],
            
            # Enhanced procedure patterns
            "PROCEDURE_SECTION": [
                r'(?:[\d\.]+\s+)?(?P<name>.*?)\s+procedure\s*\n',
                r'(?:[\d\.]+\s+)?Procedure\s+for\s+(?P<name>.*?)\n',
                r'(?:[\d\.]+\s+)?(?P<name>.*?)\s+procedures?\s+description\s*\n'
            ],
            
            "PROCEDURE_STEP": [
                r'(?:[\d\.]+\s+)?(?P<step_num>\d+)[).]\s*(?P<content>.*?)(?=\n\d+[).]|\n\n|$)',
                r'(?:Step|step)\s+(?P<step_num>\d+)[:.]\s*(?P<content>.*?)(?=\n\s*Step|step\s+\d+[:.]\s*|\n\n|$)',
                r'(?P<content>(?:The|Then|Next|Finally)\s+.*?)(?=\n\n|$)'
            ],
            
            "ACTOR": [
                r'(?:the\s+)?(?P<actor>UE|AMF|SMF|MME|gNB|Network|AUSF|SEAF|UDM)\s+(?:shall|should|must|will|may)',
                r'(?P<actor>UE|AMF|SMF|MME|gNB|Network|AUSF|SEAF|UDM)(?:\s+is|\s+has|\s+performs?)'
            ],
            
            "ACTION": [
                r'(?:shall|should|must|will|may)\s+(?P<action>(?:send|receive|transmit|forward|process|verify|check|determine|establish|release|initiate).*?)(?=\.|$)',
                r'(?:performs?|executes?|initiates?)\s+(?P<action>.*?)(?=\.|$)'
            ],
            
            "CONDITION": [
                r'[Ii]f\s+(?P<condition>.*?),\s+(?:then\s+)?(?P<true_path>.*?)(?:\.\s+(?:[Oo]therwise|[Ee]lse),?\s+(?P<false_path>.*?))?(?=\.|$)',
                r'[Ww]hen\s+(?P<condition>.*?),\s+(?:then\s+)?(?P<true_path>.*?)(?:\.\s+(?:[Oo]therwise|[Ee]lse),?\s+(?P<false_path>.*?))?(?=\.|$)'
            ],
            
            "PARAMETER": [
                r'(?:with|using|containing|including)\s+(?:the\s+)?(?P<param>[\w\-]+(?:\s+[\w\-]+)*?)(?=\s+(?:and|or|,|\.|$))',
                r'(?:sets?|includes?|contains?)\s+(?:the\s+)?(?P<param>[\w\-]+(?:\s+[\w\-]+)*?)(?=\s+(?:and|or|,|\.|$))'
            ]
        }
        
        # Compile all patterns
        self.compiled_patterns = {
            entity_type: [re.compile(pattern, re.MULTILINE | re.IGNORECASE) 
                         for pattern in patterns]
            for entity_type, patterns in self.patterns.items()
        }

        # Known 5G Registration procedure steps
        self.registration_steps = {
            "Initial Registration": [
                {
                    "step": 1,
                    "actor": "UE",
                    "action": "sends Registration Request",
                    "parameters": ["SUCI", "5GS registration type", "5GMM capability"],
                    "description": "UE initiates registration by sending Registration Request with identity and capabilities"
                },
                {
                    "step": 2,
                    "actor": "AMF",
                    "action": "initiates Primary Authentication",
                    "parameters": ["5G-AKA", "SUCI", "SUPI"],
                    "description": "AMF starts authentication procedure to verify UE identity"
                },
                {
                    "step": 3,
                    "actor": "AMF",
                    "action": "sends Security Mode Command",
                    "parameters": ["UE security capability", "5G NAS security algorithms"],
                    "description": "AMF activates NAS security with the UE"
                },
                {
                    "step": 4,
                    "actor": "UE",
                    "action": "sends Security Mode Complete",
                    "parameters": ["IMEISV", "NAS-MAC"],
                    "description": "UE confirms security activation and provides equipment identity if requested"
                },
                {
                    "step": 5,
                    "actor": "AMF",
                    "action": "sends Registration Accept",
                    "parameters": ["5G-GUTI", "Registration result", "Allowed NSSAI"],
                    "description": "AMF accepts registration and provides temporary identity and allowed services"
                },
                {
                    "step": 6,
                    "actor": "UE",
                    "action": "sends Registration Complete",
                    "parameters": ["5G-GUTI"],
                    "description": "UE acknowledges registration completion and new temporary identity"
                }
            ]
        }

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract all entities from text using compiled patterns"""
        entities = {entity_type: [] for entity_type in self.patterns.keys()}
        
        for entity_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(text)
                for match in matches:
                    entity = match.group().strip()
                    if entity not in entities[entity_type]:
                        entities[entity_type].append(entity)
        
        return entities

    def extract_procedure_flow(self, text: str) -> List[Dict]:
        """Extract procedure flows from text dynamically."""
        procedures = []
        
        # Find procedure sections
        for pattern in self.compiled_patterns["PROCEDURE_SECTION"]:
            for match in pattern.finditer(text):
                procedure_name = match.group("name").strip()
                # Get the text until the next procedure section or end of text
                start_idx = match.end()
                next_match = pattern.search(text[start_idx:])
                end_idx = start_idx + next_match.start() if next_match else len(text)
                procedure_text = text[start_idx:end_idx]
                
                steps = self._extract_steps(procedure_text)
                if steps:
                    procedures.append({
                        "name": procedure_name,
                        "steps": steps
                    })
        
        return procedures

    def _extract_steps(self, procedure_text: str) -> List[Dict]:
        """Extract steps from procedure text."""
        steps = []
        current_step = None
        step_number = 1
        
        # First try to find numbered steps
        for pattern in self.compiled_patterns["PROCEDURE_STEP"]:
            for match in pattern.finditer(procedure_text):
                step_content = match.group("content").strip()
                step_num = int(match.group("step_num")) if "step_num" in match.groupdict() else step_number
                
                step = self._process_step_content(step_content, step_num)
                if step:
                    steps.append(step)
                    step_number = step_num + 1
        
        # If no numbered steps found, try to extract sequential steps
        if not steps:
            sentences = re.split(r'[.!?]\s+', procedure_text)
            for sentence in sentences:
                if sentence.strip():
                    step = self._process_step_content(sentence.strip(), step_number)
                    if step:
                        steps.append(step)
                        step_number += 1
        
        return steps

    def _process_step_content(self, content: str, step_number: int) -> Dict:
        """Process step content to extract actor, action, parameters, and conditions."""
        step = {
            "step_number": step_number,
            "description": content,
            "actor": None,
            "action": None,
            "parameters": [],
            "conditions": []
        }
        
        # Extract actor
        for pattern in self.compiled_patterns["ACTOR"]:
            match = pattern.search(content)
            if match:
                step["actor"] = match.group("actor")
                break
        
        # Extract action
        for pattern in self.compiled_patterns["ACTION"]:
            match = pattern.search(content)
            if match:
                step["action"] = match.group("action").strip()
                break
        
        # Extract parameters
        for pattern in self.compiled_patterns["PARAMETER"]:
            for match in pattern.finditer(content):
                param = match.group("param").strip()
                if param and param not in step["parameters"]:
                    step["parameters"].append(param)
        
        # Extract conditions
        for pattern in self.compiled_patterns["CONDITION"]:
            for match in pattern.finditer(content):
                condition = {
                    "condition": match.group("condition").strip(),
                    "true_path": match.group("true_path").strip() if match.group("true_path") else None,
                    "false_path": match.group("false_path").strip() if match.group("false_path") else None
                }
                step["conditions"].append(condition)
        
        return step if (step["actor"] or step["action"]) else None

    def extract_state_transitions(self, text: str) -> List[Tuple[str, str, List[str]]]:
        """Extract state transitions and their conditions"""
        transitions = []
        state_pattern = r'(?:EMM|MM|GMM|5GMM)-[A-Z-]+'
        
        # Find all states in the text
        states = re.findall(state_pattern, text)
        
        # Look for transitions between states
        for i, state1 in enumerate(states):
            if i + 1 < len(states):
                state2 = states[i + 1]
                
                # Find conditions between these states
                start_idx = text.find(state1)
                end_idx = text.find(state2)
                if start_idx != -1 and end_idx != -1:
                    between_text = text[start_idx:end_idx]
                    
                    # Extract conditions
                    conditions = []
                    condition_pattern = r'if\s+([^,\.]+)'
                    condition_matches = re.finditer(condition_pattern, between_text)
                    for match in condition_matches:
                        conditions.append(match.group(1).strip())
                    
                    transitions.append((state1, state2, conditions))
        
        return transitions

    def process_document(self, file_path: str) -> Dict:
        """Process a single document and extract all relevant information"""
        try:
            # Use preprocess_pdfs to read and clean the document in chunks
            document_chunks = read_pdfs_from_directory(
                directory=os.path.dirname(file_path),
                pattern=os.path.basename(file_path),
                chunk_size=1000  # Adjust chunk size as needed
            )
            
            if not document_chunks:
                logger.error(f"No content extracted from {file_path}")
                return None
                
            # Initialize combined results
            combined_result = {
                "file_path": file_path,
                "entities": {entity_type: [] for entity_type in self.patterns.keys()},
                "procedure_flows": [],
                "state_transitions": []
            }
            
            # Process each chunk
            for chunk in document_chunks:
                text = chunk["content"]
                
                # Extract entities from chunk
                chunk_entities = self.extract_entities(text)
                for entity_type, entities in chunk_entities.items():
                    combined_result["entities"][entity_type].extend(entities)
                
                # Extract procedure flows from chunk
                chunk_procedures = self.extract_procedure_flow(text)
                combined_result["procedure_flows"].extend(chunk_procedures)
                
                # Extract state transitions from chunk
                chunk_transitions = self.extract_state_transitions(text)
                combined_result["state_transitions"].extend(chunk_transitions)
            
            # Remove duplicates
            for entity_type in combined_result["entities"]:
                combined_result["entities"][entity_type] = list(set(combined_result["entities"][entity_type]))
            
            # Remove duplicate procedures by name
            seen_procedures = set()
            unique_procedures = []
            for proc in combined_result["procedure_flows"]:
                if proc["name"] not in seen_procedures:
                    seen_procedures.add(proc["name"])
                    unique_procedures.append(proc)
            combined_result["procedure_flows"] = unique_procedures
            
            # Remove duplicate transitions
            combined_result["state_transitions"] = list(set(tuple(trans) for trans in combined_result["state_transitions"]))
            
            return combined_result
                
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return None

def save_extracted_data(processed_files: List[Dict], output_file: str = "extracted_data.json"):
    """Save extracted data to a JSON file for inspection."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_files, f, indent=2, ensure_ascii=False)
        console.print(f"[green]Extracted data saved to {output_file}[/green]")
    except Exception as e:
        console.print(f"[red]Error saving extracted data: {str(e)}[/red]")

def main():
    console.print("[bold blue]Starting 3GPP NAS Entity Extraction[/bold blue]")
    
    # Initialize extractor
    extractor = ThreeGPPEntityExtractor()
    
    # Load cache
    cache = load_cache()
    
    # Process PDF files
    pdf_dir = "data"
    processed_files = []
    
    # Get list of PDF files
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    console.print(f"[green]Found {len(pdf_files)} PDF files to process[/green]")
    
    with tqdm(total=len(pdf_files), desc="Processing PDFs") as pbar:
        for file_name in pdf_files:
            file_path = os.path.join(pdf_dir, file_name)
            file_hash = get_file_hash(file_path)
            
            # Check if file was already processed
            if file_hash in cache:
                console.print(f"[yellow]Skipping already processed file: {file_name}[/yellow]")
                processed_files.append(cache[file_hash])
                pbar.update(1)
                continue
            
            console.print(f"[green]Processing file: {file_name}[/green]")
            
            try:
                # Process document in chunks
                documents = read_pdfs_from_directory(
                    directory=pdf_dir,
                    pattern=file_name
                )
                
                if not documents:
                    console.print(f"[red]No content extracted from {file_name}[/red]")
                    pbar.update(1)
                    continue
                
                # Initialize combined results
                combined_result = {
                    "file_path": file_path,
                    "entities": {entity_type: [] for entity_type in extractor.patterns.keys()},
                    "procedure_flows": [],
                    "state_transitions": []
                }
                
                # Process each chunk
                for doc in documents:  # Process ALL chunks, not just the first one
                    text = doc["content"]
                    chunk_num = doc.get("chunk_number", 0)
                    total_chunks = doc.get("total_chunks", 0)
                    
                    console.print(f"  Processing chunk {chunk_num}/{total_chunks}")
                    
                    # Extract information from this chunk
                    entities = extractor.extract_entities(text)
                    procedure_flows = extractor.extract_procedure_flow(text)
                    state_transitions = extractor.extract_state_transitions(text)
                    
                    # Merge entities
                    for entity_type, entity_list in entities.items():
                        combined_result["entities"][entity_type].extend(entity_list)
                    
                    # Merge procedure flows
                    combined_result["procedure_flows"].extend(procedure_flows)
                    
                    # Merge state transitions
                    combined_result["state_transitions"].extend(state_transitions)
                
                # Remove duplicates
                for entity_type in combined_result["entities"]:
                    combined_result["entities"][entity_type] = list(set(combined_result["entities"][entity_type]))
                
                # Remove duplicate procedures by name
                seen_procedures = set()
                unique_procedures = []
                for proc in combined_result["procedure_flows"]:
                    if proc["name"] not in seen_procedures:
                        seen_procedures.add(proc["name"])
                        unique_procedures.append(proc)
                combined_result["procedure_flows"] = unique_procedures
                
                # Remove duplicate transitions
                combined_result["state_transitions"] = list(set(tuple(trans) for trans in combined_result["state_transitions"]))
                
                # Print extraction summary for this file
                console.print(f"\n[cyan]Extraction Summary for {file_name}:[/cyan]")
                console.print("Entities found:")
                for entity_type, entities in combined_result["entities"].items():
                    if entities:
                        console.print(f"  {entity_type}: {len(entities)} entities")
                console.print(f"Procedures found: {len(combined_result['procedure_flows'])}")
                console.print(f"State transitions found: {len(combined_result['state_transitions'])}")
                
                # Store in cache and add to processed files
                if (any(entities for entities in combined_result["entities"].values()) or 
                    combined_result["procedure_flows"] or 
                    combined_result["state_transitions"]):
                    cache[file_hash] = combined_result
                    processed_files.append(combined_result)
                    save_cache(cache)
                
            except Exception as e:
                console.print(f"[red]Error processing {file_name}: {str(e)}[/red]")
                import traceback
                console.print(f"[red]Traceback:[/red]\n{traceback.format_exc()}")
            
            pbar.update(1)
    
    # Save extracted data for inspection
    save_extracted_data(processed_files)
    
    console.print("[green]PDF processing completed. Storing results in Neo4j...[/green]")
    
    # Store results in Neo4j
    try:
        graph = KnowledgeGraph(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD
        )
        
        # Log what we're about to store
        for file_result in processed_files:
            console.print("\n[bold cyan]Extracted Data Summary:[/bold cyan]")
            console.print(f"[green]Entities found:[/green]")
            for entity_type, entities in file_result["entities"].items():
                console.print(f"  {entity_type}: {len(entities)} entities")
            
            console.print(f"\n[green]Procedures found:[/green] {len(file_result['procedure_flows'])}")
            for proc in file_result["procedure_flows"]:
                console.print(f"  - {proc['name']}: {len(proc['steps'])} steps")
            
            console.print(f"\n[green]State Transitions found:[/green] {len(file_result['state_transitions'])}")
        
        with tqdm(total=len(processed_files), desc="Storing in Neo4j") as pbar:
            for file_result in processed_files:
                # Store entities
                for entity_type, entities in file_result["entities"].items():
                    for entity in entities:
                        graph.create_entity(entity, entity_type)
                
                # Store procedure flows (fixed to handle list structure)
                for procedure in file_result["procedure_flows"]:
                    graph.create_procedure(procedure["name"], procedure["steps"])
                
                # Store state transitions
                for from_state, to_state, conditions in file_result["state_transitions"]:
                    graph.create_state_transition(from_state, to_state, conditions)
                
                pbar.update(1)
        
        console.print("[bold green]Successfully stored results in Neo4j[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error storing results in Neo4j: {str(e)}[/bold red]")
        # Add more detailed error information
        import traceback
        console.print(f"[red]Full error traceback:[/red]\n{traceback.format_exc()}")
    
    console.print("[bold blue]Entity extraction completed[/bold blue]")

if __name__ == "__main__":
    main() 