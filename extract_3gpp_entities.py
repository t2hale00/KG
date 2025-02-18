import os
import re
import json
from typing import List, Dict, Tuple, Set
from pypdf import PdfReader
from neo4j import GraphDatabase
from preprocess_pdfs import read_pdfs_from_directory
from rich.console import Console
from tqdm import tqdm
import hashlib
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from nltk.tree import Tree
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()  # This will load the variables from the .env file

console = Console()
CACHE_FILE = "processed_docs_cache.json"

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('chunkers/maxent_ne_chunker')
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')

class NLPEntityExtractor:
    def __init__(self):
        self.procedure_indicators = {
            'start': [
                'procedure', 'general', 'overview', 'description',
                'begin', 'start', 'initialize', 'trigger',
                # 3GPP common procedure starts
                'registration procedure', 'handover procedure',
                'authentication procedure', 'service request procedure',
                'pdu session establishment', 'deregistration procedure'
            ],
            'action': [
                # General actions
                'perform', 'execute', 'send', 'receive', 'check', 'verify',
                # 3GPP specific actions
                'transmit', 'forward', 'initiate', 'establish', 'release',
                'allocate', 'authenticate', 'authorize', 'validate'
            ],
            'condition': [
                # General conditions
                'if', 'when', 'upon', 'after', 'before', 'while', 'unless',
                # 3GPP specific conditions
                'in case', 'provided that', 'subject to', 'as specified in',
                'according to', 'based on', 'depending on'
            ],
            'end': [
                # General endings
                'end', 'complete', 'finish', 'terminate',
                # 3GPP specific endings
                'procedure complete', 'procedure ends', 'completion',
                'successful completion', 'unsuccessful completion'
            ]
        }
        
        self.procedure_patterns = [
            # Standard numbered steps
            r'(?P<step_num>\d+\.|[a-z]\)|\d+\))\s*(?P<content>.*?)(?=\d+\.|[a-z]\)|\d+\)|$)',
            
            # Explicit step markers
            r'Step\s+(?P<step_num>\d+)[\.:]\s*(?P<content>.*?)(?=Step|$)',
            
            # 3GPP specific patterns
            r'(?P<content>The\s+(?:UE|AMF|network|MME|eNB|gNB)\s+shall\s+.*?)(?=\.|$)',
            r'(?P<content>If\s+.*?\s+then\s+.*?)(?=\.|$)',
            
            # Procedure references
            r'(?P<content>as specified in clause\s+[\d\.]+\s*.*?)(?=\.|$)',
            
            # State transitions
            r'(?P<content>(?:UE|AMF|network)\s+(?:enters|moves to|transitions to)\s+.*?\s+state)(?=\.|$)',
            
            # Message flows
            r'(?P<content>(?:UE|AMF|network)\s+(?:sends|receives)\s+.*?\s+message)(?=\.|$)',
            
            # Conditional procedures
            r'(?P<content>In\s+case\s+of\s+.*?,\s+.*?)(?=\.|$)',
            
            # Requirements
            r'(?P<content>The\s+.*?\s+shall\s+.*?)(?=\.|$)',
            r'(?P<content>The\s+.*?\s+may\s+.*?)(?=\.|$)',
            r'(?P<content>The\s+.*?\s+should\s+.*?)(?=\.|$)'
        ]

        # Define entity types for NLTK
        self.entity_types = {
            'NETWORK_ELEMENT': 5,  # Priority level
            'PROTOCOL': 4,
            'MESSAGE': 4,
            'STATE': 3,
            'EVENT': 3,
            'ACTION': 2,
            'PARAMETER': 2,
            'REFERENCE': 1
        }
        
        # Expanded domain mappings for 3GPP entities
        self.domain_mappings = {
            # Network Elements
            'UE': 'NETWORK_ELEMENT',
            'AMF': 'NETWORK_ELEMENT',
            'SMF': 'NETWORK_ELEMENT',
            'UPF': 'NETWORK_ELEMENT',
            'MME': 'NETWORK_ELEMENT',
            'ENB': 'NETWORK_ELEMENT',
            'GNB': 'NETWORK_ELEMENT',
            'PCF': 'NETWORK_ELEMENT',
            'AUSF': 'NETWORK_ELEMENT',
            'UDM': 'NETWORK_ELEMENT',
            
            # Protocols
            'RRC': 'PROTOCOL',
            'NAS': 'PROTOCOL',
            'NGAP': 'PROTOCOL',
            'HTTP': 'PROTOCOL',
            'DIAMETER': 'PROTOCOL',
            
            # States
            'CM-IDLE': 'STATE',
            'CM-CONNECTED': 'STATE',
            'RM-REGISTERED': 'STATE',
            'RM-DEREGISTERED': 'STATE',
            'RRC_IDLE': 'STATE',
            'RRC_CONNECTED': 'STATE',
            
            # Common procedures
            'REGISTRATION': 'PROCEDURE',
            'DEREGISTRATION': 'PROCEDURE',
            'HANDOVER': 'PROCEDURE',
            'SERVICE_REQUEST': 'PROCEDURE',
            'PDU_SESSION_ESTABLISHMENT': 'PROCEDURE',
            
            # Messages
            'REGISTRATION_REQUEST': 'MESSAGE',
            'REGISTRATION_ACCEPT': 'MESSAGE',
            'REGISTRATION_COMPLETE': 'MESSAGE',
            'REGISTRATION_REJECT': 'MESSAGE',
            'DEREGISTRATION_REQUEST': 'MESSAGE',
            'DEREGISTRATION_ACCEPT': 'MESSAGE',
            'SERVICE_REQUEST': 'MESSAGE',
            'SERVICE_ACCEPT': 'MESSAGE',
            'SERVICE_REJECT': 'MESSAGE'
        }

        # Add specific procedure categories
        self.procedure_categories = {
            'MOBILITY': [
                'registration', 'deregistration', 'handover', 'cell reselection',
                'tracking area update', 'location update', 'paging'
            ],
            'SESSION': [
                'pdu session', 'bearer setup', 'qos flow', 'connection setup',
                'context setup', 'resource setup'
            ],
            'SECURITY': [
                'authentication', 'security mode', 'key agreement', 'integrity',
                'encryption', 'identification'
            ],
            'ERROR_HANDLING': [
                'error', 'failure', 'reject', 'abort', 'recovery',
                'fallback', 'retry'
            ]
        }

        # Add validation rules
        self.validation_rules = {
            'NETWORK_ELEMENT': {
                'required_properties': ['id', 'type', 'role'],
                'allowed_relationships': ['CONNECTS_TO', 'SERVES', 'CONTROLS']
            },
            'PROCEDURE': {
                'required_properties': ['category', 'trigger', 'outcome'],
                'allowed_relationships': ['INVOLVES', 'TRIGGERS', 'FOLLOWS']
            },
            'MESSAGE': {
                'required_properties': ['direction', 'protocol'],
                'allowed_relationships': ['SENT_BY', 'RECEIVED_BY', 'PART_OF']
            },
            'STATE': {
                'required_properties': ['entity_type', 'conditions'],
                'allowed_relationships': ['TRANSITIONS_TO', 'TRIGGERED_BY']
            }
        }

        # Add cross-specification reference patterns
        self.cross_ref_patterns = {
            'TS_REFERENCE': r'(?:3GPP\s+)?TS\s+\d{2}\.\d{3}(?:\s+\[(\d+)\])?',
            'CLAUSE_REFERENCE': r'clause\s+\d+(?:\.\d+)*(?:\s+of\s+(?:3GPP\s+)?TS\s+\d{2}\.\d{3})?',
            'REQUIREMENT_REFERENCE': r'R\d+(?:\.\d+)*(?:\s+in\s+(?:3GPP\s+)?TS\s+\d{2}\.\d{3})?',
            'FIGURE_REFERENCE': r'Figure\s+\d+(?:\.\d+)*(?:-\d+)?',
            'TABLE_REFERENCE': r'Table\s+\d+(?:\.\d+)*(?:-\d+)?'
        }

    def extract_procedures(self, text: str) -> List[Dict]:
        """Extract procedural steps and their structure from text."""
        procedures = []
        sentences = sent_tokenize(text)
        
        current_procedure = None
        for sentence in sentences:
            # Check if this starts a new procedure
            if any(word in sentence.lower() for word in self.procedure_indicators['start']):
                if current_procedure:
                    procedures.append(current_procedure)
                current_procedure = {
                    'type': 'procedure',
                    'steps': [],
                    'conditions': [],
                    'description': sentence
                }
                continue
                
            if current_procedure:
                # Extract steps
                for pattern in self.procedure_patterns:
                    matches = re.finditer(pattern, sentence)
                    for match in matches:
                        step_content = match.group('content').strip()
                        step = {
                            'text': step_content,
                            'type': self._determine_step_type(step_content),
                            'entities': self._extract_step_entities(step_content)
                        }
                        current_procedure['steps'].append(step)
                
                # Extract conditions
                if any(word in sentence.lower() for word in self.procedure_indicators['condition']):
                    current_procedure['conditions'].append({
                        'text': sentence,
                        'entities': self._extract_step_entities(sentence)
                    })
                
                # Check if procedure ends
                if any(word in sentence.lower() for word in self.procedure_indicators['end']):
                    procedures.append(current_procedure)
                    current_procedure = None
        
        # Add any remaining procedure
        if current_procedure:
            procedures.append(current_procedure)
            
        return procedures

    def _determine_step_type(self, text: str) -> str:
        """Determine the type of a procedural step."""
        text_lower = text.lower()
        if any(word in text_lower for word in self.procedure_indicators['condition']):
            return 'condition'
        elif any(word in text_lower for word in self.procedure_indicators['action']):
            return 'action'
        return 'information'

    def _extract_step_entities(self, text: str) -> List[Dict]:
        """Extract named entities from a step using NLTK and custom rules."""
        entities = {}  # Use dict for deduplication
        
        # Tokenize and tag the text
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        
        # Extract technical terms (uppercase words/acronyms)
        for token, pos in pos_tags:
            if token.isupper() and len(token) > 1:
                entity_type = self._determine_entity_type(token)
                if entity_type:
                    entity_key = (token.lower(), entity_type)
                    entities[entity_key] = {
                        'text': token,
                        'type': entity_type,
                        'source': 'technical_term',
                        'confidence': 1.0
                    }
        
        # Extract phrases using custom patterns
        self._extract_custom_entities(text, entities)
        
        # Extract references and cross-references
        self._extract_references(text, entities)
        
        # Validate and enrich entities
        validated_entities = []
        for entity in entities.values():
            validated_entity = self._validate_entity(entity)
            validated_entities.append(validated_entity)
        
        return validated_entities

    def _determine_entity_type(self, token: str) -> str:
        """Determine the type of an entity based on domain mappings and patterns."""
        # Check domain mappings first
        if token in self.domain_mappings:
            return self.domain_mappings[token]
        
        # Check patterns for network functions
        if re.match(r'^[A-Z]+F$', token):  # AMF, SMF, UPF, etc.
            return 'NETWORK_ELEMENT'
        if re.match(r'^[A-Z]+-[A-Z]+$', token):  # N1, N2, etc.
            return 'PROTOCOL'
        if '_REQUEST' in token or '_RESPONSE' in token or '_ACCEPT' in token:
            return 'MESSAGE'
        if '_STATE' in token or token.endswith(('_IDLE', '_CONNECTED')):
            return 'STATE'
        
        return None

    def _extract_custom_entities(self, text: str, entities: Dict) -> None:
        """Extract entities using custom patterns."""
        custom_patterns = {
            # Network Elements
            r'\b(?:User Equipment|UE|Mobile Station|MS)\b': 'NETWORK_ELEMENT',
            r'\b(?:g?[eE]N[Bb]|[RB]AN|NG-RAN)\b': 'NETWORK_ELEMENT',
            
            # States
            r'\b(?:CM|RM|RRC|EMM|ECM|PMM)[\s_-]+(?:IDLE|CONNECTED|REGISTERED|DEREGISTERED)\b': 'STATE',
            
            # Messages
            r'\b(?:[A-Z]+_[A-Z]+_(?:REQUEST|RESPONSE|ACCEPT|REJECT|COMPLETE))\b': 'MESSAGE',
            
            # Parameters
            r'\b(?:IE|Information Element)s?\b': 'PARAMETER',
            
            # Actions
            r'\b(?:shall|should|may|must)\s+(?:[a-z]+(?:\s+[a-z]+)*)\b': 'ACTION'
        }
        
        for pattern, entity_type in custom_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_text = match.group()
                entity_key = (entity_text.lower(), entity_type)
                if entity_key not in entities:
                    entities[entity_key] = {
                        'text': entity_text,
                        'type': entity_type,
                        'source': 'custom_rules',
                        'confidence': 1.0,
                        'context': text[max(0, match.start()-50):min(len(text), match.end()+50)]
                    }

    def _extract_references(self, text: str, entities: Dict) -> None:
        """Extract references and cross-references."""
        reference_patterns = {
            r'\bclause\s+[\d\.]+\b': 'REFERENCE',
            r'\b(?:TS|TR)\s+\d+\.\d+(?:\.\d+)*\b': 'REFERENCE',
            r'\[[\d\w-]+\]': 'REFERENCE'
        }
        
        for pattern, entity_type in reference_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entity_text = match.group()
                entity_key = (entity_text.lower(), entity_type)
                if entity_key not in entities:
                    entities[entity_key] = {
                        'text': entity_text,
                        'type': entity_type,
                        'source': 'reference',
                        'confidence': 1.0,
                        'context': text[max(0, match.start()-50):min(len(text), match.end()+50)]
                    }

    def _validate_entity(self, entity: Dict) -> Dict:
        """Validate and enrich entity based on validation rules."""
        entity_type = entity.get('type')
        rules = self.validation_rules.get(entity_type, {})
        
        # Check required properties
        for prop in rules.get('required_properties', []):
            if prop not in entity:
                entity[prop] = self._infer_property(entity, prop)
        
        # Add allowed relationships
        entity['allowed_relationships'] = rules.get('allowed_relationships', [])
        
        return entity

    def _infer_property(self, entity: Dict, property_name: str) -> str:
        """Infer missing required properties based on context and rules."""
        text = entity.get('text', '').lower()
        context = entity.get('context', '').lower()
        
        if property_name == 'id':
            return f"{entity['type']}_{hash(text) % 10000}"
        
        elif property_name == 'category':
            return self._categorize_procedure(text)
        
        elif property_name == 'direction':
            if any(word in context for word in ['sends', 'transmits', 'initiates']):
                return 'OUTGOING'
            elif any(word in context for word in ['receives', 'accepts']):
                return 'INCOMING'
            return 'UNKNOWN'
        
        elif property_name == 'protocol':
            protocols = ['NAS', 'NGAP', 'RRC', 'HTTP2']
            for protocol in protocols:
                if protocol.lower() in context:
                    return protocol
            return 'UNKNOWN'
        
        return 'UNKNOWN'

    def _categorize_procedure(self, text: str) -> str:
        """Categorize a procedure based on its description."""
        text_lower = text.lower()
        
        for category, keywords in self.procedure_categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return 'GENERAL'

    def _merge_overlapping_entities(self, entities: List[Dict]) -> List[Dict]:
        """Merge overlapping entities based on text span and type priority."""
        if not entities:
            return entities
            
        # Sort entities by start position and length
        sorted_entities = sorted(
            entities,
            key=lambda x: (x['start'], -len(x['text']))
        )
        
        merged = []
        current = sorted_entities[0]
        
        for next_entity in sorted_entities[1:]:
            # Check for overlap
            if (next_entity['start'] >= current['start'] and 
                next_entity['start'] <= current['start'] + len(current['text'])):
                # Decide which entity to keep based on type priority and confidence
                if (self._get_entity_priority(next_entity['type']) > 
                    self._get_entity_priority(current['type']) or
                    next_entity['confidence'] > current['confidence']):
                    current = next_entity
            else:
                merged.append(current)
                current = next_entity
        
        merged.append(current)
        return merged

    def _get_entity_priority(self, entity_type: str) -> int:
        """Get priority for entity type (higher number = higher priority)."""
        return self.entity_types.get(entity_type, 0)

    def extract_metadata(self, text: str) -> Dict:
        """Extract metadata from text using NLP techniques."""
        metadata = {
            'version_info': [],
            'references': [],
            'section_headers': [],
            'technical_terms': set()
        }
        
        sentences = sent_tokenize(text)
        
        # Extract version information
        version_pattern = r'(?:version|release)\s*(?P<version>[\d\.]+)'
        versions = re.finditer(version_pattern, text, re.IGNORECASE)
        metadata['version_info'].extend([m.group('version') for m in versions])
        
        # Extract references
        ref_pattern = r'\[[\d\w-]+\]|\b\d+\.\d+\.\d+\b'
        refs = re.finditer(ref_pattern, text)
        metadata['references'].extend([m.group() for m in refs])
        
        # Extract section headers
        for sent in sentences:
            if sent.strip().isupper() or re.match(r'\d+\.\d+\s+[A-Z]', sent):
                metadata['section_headers'].append(sent.strip())
        
        # Extract technical terms
        for token in text:
            if token.isupper() and len(token) > 1:
                metadata['technical_terms'].add(token)
        
        return metadata

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
        self.nlp_extractor = NLPEntityExtractor()
        
        # Core patterns for Mobility Management analysis
        self.patterns = {
            "UE": [
                r'\b(?:UE|User Equipment)\b',
                r'\b(?:Mobile Station|MS)\b',
                r'\b(?:UE|MS)\s+(?:ID|Identifier|Identity)\b',
                r'\b(?:UE|MS)\s+(?:Status|State|Mode)\b',
                r'\b(?:UE|MS)\s+(?:Security Context|GUTI)\b'
            ],
            "AMF": [
                r'\b(?:AMF|Access and Mobility Management Function)\b',
                r'\b(?:AMF)\s+(?:ID|Identifier|Region|Set)\b',
                r'\b(?:AMF)\s+(?:Status|State|Capacity)\b',
                r'\b(?:AMF)\s+(?:Selection|Reselection)\b',
                r'\b(?:AMF)\s+(?:Load Level|Served Areas)\b'
            ],
            "REGISTRATION": [
                r'\b(?:Initial|Periodic|Emergency)\s+Registration\b',
                r'\b(?:Registration|Update)\s+(?:Request|Accept|Reject|Complete)\b',
                r'\b(?:Registration)\s+(?:Area|Timer|Status)\b',
                r'\b(?:Registration)\s+(?:Type|Procedure|Flow)\b',
                r'\b(?:Follow-on Request|Update Type)\b'
            ],
            "DEREGISTRATION": [
                # Deregistration Patterns
                r'\b(?:UE|Network)-initiated\s+Deregistration\b',
                r'\b(?:Deregistration)\s+(?:Request|Accept|Reject)\b',
                r'\b(?:Deregistration)\s+(?:Type|Procedure|Flow)\b',
                r'\b(?:Implicit|Explicit)\s+Deregistration\b'
            ],
            "CONNECTION": [
                r'\b(?:RRC|CM|RM)\s+(?:Connected|Idle|Inactive)\b',
                r'\b(?:Connection)\s+(?:Setup|Release|Modify)\b',
                r'\b(?:Connection)\s+(?:State|Status|Type|Quality)\b',
                r'\b(?:PDU|Bearer)\s+(?:Session|Connection)\b',
                r'\b(?:Establishment|Release)\s+Cause\b'
            ],
            "NETWORK_AREA": [
                r'\b(?:Tracking|Registration|Location)\s+Area\b',
                r'\b(?:TA|RA|LA)\s+(?:Code|Identity|List)\b',
                r'\b(?:Cell|Sector|Coverage)\s+(?:ID|Area)\b',
                r'\b(?:PLMN|Network)\s+(?:ID|Code|Area)\b',
                r'\b(?:Access Type|Coverage Level|Congestion Level)\b'
            ],
            "MOBILITY_EVENT": [
                r'\b(?:Handover|Cell Reselection|Cell Change)\b',
                r'\b(?:Mobility)\s+(?:Event|Trigger|Update)\b',
                r'\b(?:Location|Area)\s+(?:Update|Change)\b',
                r'\b(?:Inter|Intra)-(?:System|RAT|Cell)\s+(?:Handover|Change)\b',
                r'\b(?:Source|Target)\s+(?:Area|Cell)\b'
            ],
            "PROCEDURE_TYPE": [
                # Procedure Type Patterns
                r'\b(?:Initial|Periodic|Emergency)\s+(?:Registration|Update)\b',
                r'\b(?:Service|Tracking Area)\s+(?:Request|Update)\b',
                r'\b(?:Authentication|Security Mode|Identity)\s+(?:Procedure)\b',
                r'\b(?:N1|N2|NAS)\s+(?:Message|Procedure|Flow)\b'
            ]
        }

        # Define relationships between entities with explicit types and descriptions
        self.relationships = {
            "REGISTERED_WITH": {
                "patterns": [
                    r'(?:UE|MS|User Equipment)\s+(?:registers?|attaches?)\s+(?:with|to)\s+(?:AMF)',
                    r'(?:registration|attach)\s+(?:request|procedure)\s+(?:to|with)\s+(?:AMF)',
                    r'(?:UE|MS)\s+(?:completes?|performs?)\s+registration\s+(?:with|to)\s+(?:AMF)'
                ],
                "properties": ["timestamp", "type", "status"],
                "description": "UE registers with AMF"
            },
            "LOCATED_IN": {
                "patterns": [
                    r'(?:UE|MS)\s+(?:located|present|residing)\s+in\s+(?:area|cell)',
                    r'(?:UE|MS)\s+(?:enters?|moves? to)\s+(?:area|cell)',
                    r'(?:UE|MS)\s+(?:position|location)\s+(?:in|within)\s+(?:area|cell)'
                ],
                "properties": ["timestamp", "entry_time", "exit_time"],
                "description": "UE is located in a network area"
            },
            "SERVES_AREA": {
                "patterns": [
                    r'(?:AMF)\s+(?:serves?|covers?)\s+(?:area|cell)',
                    r'(?:area|cell)\s+(?:served|covered)\s+by\s+(?:AMF)',
                    r'(?:AMF)\s+(?:responsible for|managing)\s+(?:area|cell)'
                ],
                "properties": ["capacity", "load_level", "status"],
                "description": "AMF serves a network area"
            },
            "INVOLVED_IN": {
                "patterns": [
                    r'(?:UE|MS)\s+(?:involved in|undergoes?)\s+(?:mobility|handover)',
                    r'(?:mobility|handover)\s+(?:event|procedure)\s+(?:for|involving)\s+(?:UE|MS)',
                    r'(?:UE|MS)\s+(?:experiences?|triggers?)\s+(?:mobility|handover)'
                ],
                "properties": ["timestamp", "result", "cause"],
                "description": "UE is involved in a mobility event"
            },
            "HAS_CONNECTION": {
                "patterns": [
                    r'(?:UE|MS)\s+(?:has|establishes?)\s+(?:connection|bearer)',
                    r'(?:connection|bearer)\s+(?:established|setup)\s+for\s+(?:UE|MS)',
                    r'(?:UE|MS)\s+(?:connection|bearer)\s+(?:state|status)'
                ],
                "properties": ["establishment_time", "status", "quality"],
                "description": "UE has an active connection"
            }
        }

        # Properties for entities based on schema
        self.property_patterns = {
            "UE_PROPERTIES": {
                "id": r'(?:UE|MS)\s+ID:\s*(\S+)',
                "status": r'(?:UE|MS)\s+status:\s*(\S+)',
                "registration_time": r'registration\s+time:\s*(\S+)',
                "connection_state": r'connection\s+state:\s*(\S+)',
                "security_context": r'security\s+context:\s*(\S+)',
                "5g_guti": r'5[Gg]-GUTI:\s*(\S+)',
                "location": r'location:\s*(\S+)'
            },
            "AMF_PROPERTIES": {
                "amf_id": r'AMF\s+ID:\s*(\S+)',
                "amf_region": r'AMF\s+region:\s*(\S+)',
                "amf_set": r'AMF\s+set:\s*(\S+)',
                "capacity": r'capacity:\s*(\d+)',
                "status": r'status:\s*(\S+)',
                "served_areas": r'served\s+areas:\s*(\S+)',
                "load_level": r'load\s+level:\s*(\d+)'
            },
            "REGISTRATION_PROPERTIES": {
                "id": r'registration\s+ID:\s*(\S+)',
                "type": r'registration\s+type:\s*(\S+)',
                "status": r'registration\s+status:\s*(\S+)',
                "timestamp": r'timestamp:\s*(\S+)',
                "result": r'result:\s*(\S+)',
                "cause": r'cause:\s*(\S+)',
                "follow_on_request": r'follow-on\s+request:\s*(\S+)',
                "update_type": r'update\s+type:\s*(\S+)'
            },
            "NETWORK_AREA_PROPERTIES": {
                "area_code": r'area\s+code:\s*(\S+)',
                "plmn_id": r'PLMN\s+ID:\s*(\S+)',
                "tac": r'TAC:\s*(\S+)',
                "cell_id": r'cell\s+ID:\s*(\S+)',
                "access_type": r'access\s+type:\s*(\S+)',
                "coverage_level": r'coverage\s+level:\s*(\S+)',
                "congestion_level": r'congestion\s+level:\s*(\d+)'
            },
            "MOBILITY_EVENT_PROPERTIES": {
                "id": r'event\s+ID:\s*(\S+)',
                "event_type": r'event\s+type:\s*(\S+)',
                "trigger": r'trigger:\s*(\S+)',
                "timestamp": r'timestamp:\s*(\S+)',
                "source_area": r'source\s+area:\s*(\S+)',
                "target_area": r'target\s+area:\s*(\S+)',
                "result": r'result:\s*(\S+)',
                "cause": r'cause:\s*(\S+)'
            },
            "CONNECTION_PROPERTIES": {
                "id": r'connection\s+ID:\s*(\S+)',
                "state": r'state:\s*(\S+)',
                "type": r'type:\s*(\S+)',
                "quality": r'quality:\s*(\S+)',
                "establishment_cause": r'establishment\s+cause:\s*(\S+)',
                "release_cause": r'release\s+cause:\s*(\S+)',
                "duration": r'duration:\s*(\S+)',
                "pdu_sessions": r'PDU\s+sessions:\s*(\S+)'
            }
        }

    def extract_entities(self, text: str) -> List[Dict]:
        """Extract entities from text using NLTK."""
        entities = []
        
        # Extract technical terms using regex patterns
        regex_entities = self._extract_regex_entities(text)
        entities.extend(regex_entities)
        
        # Extract procedures and their steps
        procedures = re.finditer(r'(?i)procedure:?\s*([^.!?\n]+)', text)
        for proc_match in procedures:
            procedure_text = proc_match.group(1).strip()
            entities.append({
                'text': procedure_text,
                'type': 'PROCEDURE'
            })
            
            # Look for steps in this procedure
            steps = re.finditer(r'(?i)step\s+(\d+):\s*([^.!?\n]+)', text)
            for step_match in steps:
                step_num = step_match.group(1)
                step_text = step_match.group(2).strip()
                entities.append({
                    'text': step_text,
                    'type': 'STEP',
                    'procedure': procedure_text,
                    'step_number': int(step_num)
                })
        
        # Extract metadata using regex patterns
        metadata_patterns = {
            'VERSION': r'(?i)version\s*(\d+(?:\.\d+)*)',
            'DATE': r'(?i)date:\s*(\d{4}-\d{2}-\d{2})',
            'AUTHOR': r'(?i)author(?:s)?:\s*([^.!?\n]+)',
            'REFERENCE': r'(?i)reference:\s*([^.!?\n]+)'
        }
        
        for entity_type, pattern in metadata_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    'text': match.group(1).strip(),
                    'type': entity_type
                })
        
        # Extract technical terms using NLTK
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            pos_tags = nltk.pos_tag(words)
            
            # Look for technical terms (nouns and noun phrases)
            i = 0
            while i < len(pos_tags):
                if pos_tags[i][1].startswith('NN'):  # If it's a noun
                    # Check for compound nouns
                    term_parts = [pos_tags[i][0]]
                    j = i + 1
                    while j < len(pos_tags) and pos_tags[j][1].startswith('NN'):
                        term_parts.append(pos_tags[j][0])
                        j += 1
                    
                    term = ' '.join(term_parts)
                    if len(term) > 2:  # Ignore very short terms
                        entities.append({
                            'text': term,
                            'type': 'TECHNICAL_TERM'
                        })
                    i = j
                else:
                    i += 1
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            key = (entity['text'], entity['type'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities

    def _extract_regex_entities(self, text: str) -> List[Dict]:
        """Extract entities using regex patterns."""
        entities = []
        
        # Define regex patterns for different entity types
        patterns = {
            'NETWORK_ELEMENT': [
                r'(?i)\b(UE|gNB|AMF|SMF|UPF|PCF|UDM|AUSF|NRF|NEF|NSSF)\b',
                r'(?i)\b(Radio\s+Access\s+Network|Core\s+Network)\b'
            ],
            'MESSAGE': [
                r'(?i)\b(Registration\s+Request|Authentication\s+Request|Security\s+Mode\s+Command)\b',
                r'(?i)\b(\w+\s+Request|\w+\s+Response|\w+\s+Command|\w+\s+Complete)\b'
            ],
            'STATE': [
                r'(?i)\b(REGISTERED|DEREGISTERED|CONNECTED|IDLE)\b',
                r'(?i)\b(\w+\s+STATE)\b'
            ],
            'EVENT': [
                r'(?i)\b(Registration|Authentication|Security\s+Mode|Service\s+Request)\b',
                r'(?i)\b(\w+\s+Event)\b'
            ],
            'PROTOCOL': [
                r'(?i)\b(NAS|RRC|NGAP|HTTP|TCP|UDP)\b',
                r'(?i)\b(\w+\s+Protocol)\b'
            ]
        }
        
        # Extract entities using patterns
        for entity_type, type_patterns in patterns.items():
            for pattern in type_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    entities.append({
                        'text': match.group(1).strip(),
                        'type': entity_type
                    })
        
        return entities

    def extract_relationships(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Extract relationships between entities in the text."""
        relationships = []
        
        # Create a mapping of entity text to entity dict for faster lookup
        entity_map = {entity['text']: entity for entity in entities}
        
        # Extract procedure steps relationships
        for entity in entities:
            if entity['type'] == 'PROCEDURE':
                procedure_text = entity['text']
                # Find steps that belong to this procedure
                for step_entity in entities:
                    if step_entity['type'] == 'STEP' and step_entity.get('procedure') == procedure_text:
                        relationships.append({
                            'source': procedure_text,
                            'target': step_entity['text'],
                            'type': 'HAS_STEP',
                            'properties': {}
                        })
        
        # Extract technical term relationships
        for entity in entities:
            if entity['type'] == 'TECHNICAL_TERM':
                term_text = entity['text']
                # Look for related terms in the same sentence
                sentences = nltk.sent_tokenize(text)
                for sentence in sentences:
                    if term_text in sentence:
                        # Find other technical terms in the same sentence
                        for other_entity in entities:
                            if (other_entity['type'] == 'TECHNICAL_TERM' and 
                                other_entity['text'] != term_text and 
                                other_entity['text'] in sentence):
                                relationships.append({
                                    'source': term_text,
                                    'target': other_entity['text'],
                                    'type': 'RELATED_TO',
                                    'properties': {'context': sentence}
                                })
        
        return relationships

def store_in_neo4j(entities, relationships):
    """Store extracted entities and relationships in Neo4j."""
    # Get Neo4j connection details from environment variables
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not neo4j_password:
        console.print("[red]❌ Error: NEO4J_PASSWORD environment variable must be set[/red]")
        return False
    
    try:
        # Connect to Neo4j
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        console.print("🔌 Connected to Neo4j database")
        
        with driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            console.print("🧹 Cleared existing data from Neo4j")
            
            # Create entities
            for entity in entities:
                cypher = """
                MERGE (e:`3GPP Entity` {name: $name, type: $type})
                """
                session.run(cypher, name=entity['text'], type=entity['type'])
            console.print(f"✅ Created {len(entities)} entities in Neo4j")
            
            # Create relationships
            for rel in relationships:
                cypher = """
                MATCH (source:`3GPP Entity` {name: $source_name})
                MATCH (target:`3GPP Entity` {name: $target_name})
                MERGE (source)-[r:`%s` {description: $description}]->(target)
                SET r += $properties
                """ % rel['type']
                
                session.run(
                    cypher,
                    source_name=rel['source'],
                    target_name=rel['target'],
                    description=rel['properties'].get('description', ''),
                    properties=rel['properties']
                )
            console.print(f"✅ Created {len(relationships)} relationships in Neo4j")
            
        driver.close()
        console.print("✅ Successfully stored data in Neo4j!")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Error connecting to Neo4j: {str(e)}[/red]")
        return False

def process_documents():
    """Process 3GPP documents and extract entities."""
    console.print("🔍 Reading and preprocessing 3GPP documents...")
    
    # Read and preprocess documents
    documents = read_pdfs_from_directory()
    
    if not documents:
        console.print("[red]❌ No documents found to process.[/red]")
        return
    
    console.print(f"✅ Successfully preprocessed {len(documents)} documents.")
    
    # Initialize entity extractor
    extractor = ThreeGPPEntityExtractor()
    
    console.print("🔍 Extracting entities and relationships...")
    
    all_entities = []
    all_relationships = []
    
    try:
        for doc in documents:
            # Process each document's content
            if 'content' not in doc:
                console.print(f"[yellow]⚠️ Skipping document {doc.get('filename', 'unknown')}: No content found[/yellow]")
                continue
                
            # Extract entities from the document
            doc_entities = extractor.extract_entities(doc['content'])
            all_entities.extend(doc_entities)
            
            # Extract relationships
            if doc_entities:
                doc_relationships = extractor.extract_relationships(doc['content'], doc_entities)
                all_relationships.extend(doc_relationships)
                console.print(f"Found {len(doc_relationships)} relationships in {doc.get('filename', 'unknown')}")
        
        # Prepare results
        results = {
            'entities': all_entities,  # Already in the correct format
            'relationships': all_relationships,
            'document_count': len(documents),
            'entity_count': len(all_entities),
            'relationship_count': len(all_relationships)
        }
        
        # Save to JSON file
        output_file = "processed_data/extracted_entities.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        console.print(f"✅ Successfully extracted {len(all_entities)} entities and {len(all_relationships)} relationships.")
        console.print(f"💾 Results saved to {output_file}")
        
        # Store in Neo4j
        console.print("💾 Storing data in Neo4j...")
        if store_in_neo4j(all_entities, all_relationships):
            console.print("✅ Data successfully stored in Neo4j!")
        else:
            console.print("[red]❌ Failed to store data in Neo4j[/red]")
        
    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")
        raise  # Re-raise the exception to see the full traceback

if __name__ == "__main__":
    process_documents()

def extract_entities_and_relationships(documents):
    """Extract entities and relationships from the documents using regex patterns."""
    extracted_entities = []
    extracted_relationships = []

    # Define regex patterns for entities and relationships
    entity_patterns = {
        "UE": r'\b(?:User Equipment|UE)\b',
        "AMF": r'\b(?:Access and Mobility Management Function|AMF)\b',
        "Registration": r'\b(?:Registration)\b',
        "Deregistration": r'\b(?:Deregistration)\b',
        "Connection": r'\b(?:Connection)\b',
        "Network Area": r'\b(?:Network Area)\b',
        "Mobility Event": r'\b(?:Mobility Event)\b',
        "Procedure Type": r'\b(?:Procedure Type)\b',
    }

    relationship_patterns = {
        "REGISTERS_WITH": r'\b(?:UE|User Equipment)\s+(?:registers|attaches)\s+(?:with|to)\s+(?:AMF)\b',
        "DEREGISTERS_FROM": r'\b(?:UE|User Equipment)\s+(?:deregisters|detaches)\s+(?:from)\s+(?:AMF)\b',
        "MOVES_TO": r'\b(?:UE|User Equipment)\s+(?:moves|enters|leaves)\s+(?:Network Area)\b',
        "HANDLES": r'\b(?:AMF)\s+(?:handles|manages)\s+(?:Mobility Event)\b',
    }

    for doc in documents:
        text = doc["text"]

        # Extract entities
        for entity_type, pattern in entity_patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                extracted_entities.append({
                    "name": match,
                    "type": entity_type,
                    "properties": {}  # Add any relevant properties if needed
                })

        # Extract relationships
        for rel_type, pattern in relationship_patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                # Assuming the relationship involves UE and AMF
                extracted_relationships.append({
                    "from": "User Equipment",  # Adjust based on actual extraction
                    "to": "Access and Mobility Management Function",  # Adjust based on actual extraction
                    "type": rel_type
                })

    return extracted_entities, extracted_relationships