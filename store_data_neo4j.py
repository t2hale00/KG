import os
import re
import json
from typing import List, Dict, Tuple, Any
from pypdf import PdfReader
from neo4j import GraphDatabase
import hashlib
import logging
from dataclasses import asdict
from datetime import datetime

# Neo4j Configuration (from environment variables)
URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

# Cache file for Neo4j state
NEO4J_CACHE_FILE = "neo4j_cache.json"

class KnowledgeGraph:
    def __init__(self, uri: str, user: str, password: str):
        """Initialize Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.logger = logging.getLogger(__name__)

    def close(self):
        """Close the Neo4j driver."""
        self.driver.close()

    def initialize_schema(self):
        """Initialize Neo4j schema with constraints and indexes."""
        with self.driver.session() as session:
            try:
                # Create constraints for unique identifiers
                constraints = [
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (s:State) REQUIRE s.name IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Event) REQUIRE e.name IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Action) REQUIRE (a.name, a.actor) IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Parameter) REQUIRE p.name IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Conditional) REQUIRE c.condition IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Step) REQUIRE (s.procedure_name, s.step_number) IS UNIQUE"
                ]
                
                # Create indexes for better performance
                indexes = [
                    "CREATE INDEX IF NOT EXISTS FOR (s:State) ON (s.type)",
                    "CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.type)",
                    "CREATE INDEX IF NOT EXISTS FOR (a:Action) ON (a.actor)",
                    "CREATE INDEX IF NOT EXISTS FOR (p:Parameter) ON (p.type)",
                    "CREATE INDEX IF NOT EXISTS FOR (s:Step) ON (s.procedure_name)"
                ]
                
                for query in constraints + indexes:
                    session.run(query)
                
                self.logger.info("Schema initialized successfully")
            except Exception as e:
                self.logger.error(f"Error initializing schema: {str(e)}")

    def create_state(self, name: str, state_type: str, description: str = None, conditions: List[str] = None):
        """Create a state node with its properties."""
        with self.driver.session() as session:
            try:
                session.execute_write(self._create_state, name, state_type, description, conditions)
            except Exception as e:
                self.logger.error(f"Error creating state {name}: {str(e)}")

    def create_event(self, name: str, event_type: str, trigger: str = None, metadata: Dict = None):
        """Create an event node with its properties."""
        with self.driver.session() as session:
            try:
                session.execute_write(self._create_event, name, event_type, trigger, metadata)
            except Exception as e:
                self.logger.error(f"Error creating event {name}: {str(e)}")

    def create_action(self, name: str, actor: str, description: str, parameters: List[str] = None):
        """Create an action node with its properties."""
        with self.driver.session() as session:
            try:
                session.execute_write(self._create_action, name, actor, description, parameters)
            except Exception as e:
                self.logger.error(f"Error creating action {name}: {str(e)}")

    def create_conditional(self, condition: str, true_path: str, false_path: str = None, parameters: List[str] = None):
        """Create a conditional node with its decision paths."""
        with self.driver.session() as session:
            try:
                session.execute_write(self._create_conditional, condition, true_path, false_path, parameters)
            except Exception as e:
                self.logger.error(f"Error creating conditional {condition}: {str(e)}")

    def create_procedure_step(self, procedure_name: str, step_number: int, action: str, 
                            actor: str, parameters: List[str] = None, conditions: List[str] = None):
        """Create a procedure step with its relationships."""
        with self.driver.session() as session:
            try:
                session.execute_write(self._create_procedure_step, procedure_name, step_number, 
                                   action, actor, parameters, conditions)
            except Exception as e:
                self.logger.error(f"Error creating step {step_number} in {procedure_name}: {str(e)}")

    @staticmethod
    def _create_state(tx, name: str, state_type: str, description: str = None, conditions: List[str] = None):
        query = """
        MERGE (s:State {name: $name})
        SET s.type = $type,
            s.description = $description,
            s.conditions = $conditions,
            s.updated_at = datetime()
        RETURN s
        """
        tx.run(query, name=name, type=state_type, description=description, conditions=conditions)

    @staticmethod
    def _create_event(tx, name: str, event_type: str, trigger: str = None, metadata: Dict = None):
        query = """
        MERGE (e:Event {name: $name})
        SET e.type = $type,
            e.trigger = $trigger,
            e.metadata = $metadata,
            e.timestamp = datetime()
        RETURN e
        """
        tx.run(query, name=name, type=event_type, trigger=trigger, metadata=metadata)

    @staticmethod
    def _create_action(tx, name: str, actor: str, description: str, parameters: List[str] = None):
        # Create action node
        action_query = """
        MERGE (a:Action {name: $name, actor: $actor})
        SET a.description = $description,
            a.updated_at = datetime()
        RETURN a
        """
        tx.run(action_query, name=name, actor=actor, description=description)
        
        # Create parameter nodes and relationships if provided
        if parameters:
            for param in parameters:
                param_query = """
                MERGE (p:Parameter {name: $param})
                WITH p
                MATCH (a:Action {name: $action_name, actor: $actor})
                MERGE (a)-[r:REQUIRES]->(p)
                RETURN p, r, a
                """
                tx.run(param_query, param=param, action_name=name, actor=actor)

    @staticmethod
    def _create_conditional(tx, condition: str, true_path: str, false_path: str = None, parameters: List[str] = None):
        # Create conditional node
        cond_query = """
        MERGE (c:Conditional {condition: $condition})
        SET c.true_path = $true_path,
            c.false_path = $false_path,
            c.updated_at = datetime()
        RETURN c
        """
        tx.run(cond_query, condition=condition, true_path=true_path, false_path=false_path)
        
        # Create parameter relationships if provided
        if parameters:
            for param in parameters:
                param_query = """
                MERGE (p:Parameter {name: $param})
                WITH p
                MATCH (c:Conditional {condition: $condition})
                MERGE (c)-[r:DEPENDS_ON]->(p)
                RETURN p, r, c
                """
                tx.run(param_query, param=param, condition=condition)

    @staticmethod
    def _create_procedure_step(tx, procedure_name: str, step_number: int, action: str, 
                             actor: str, parameters: List[str] = None, conditions: List[str] = None):
        # Create step node
        step_query = """
        MERGE (s:Step {procedure_name: $proc_name, step_number: $step_num})
        SET s.action = $action,
            s.actor = $actor,
            s.conditions = $conditions,
            s.updated_at = datetime()
        RETURN s
        """
        tx.run(step_query, proc_name=procedure_name, step_num=step_number, 
               action=action, actor=actor, conditions=conditions)
        
        # Create parameter relationships
        if parameters:
            for param in parameters:
                param_query = """
                MERGE (p:Parameter {name: $param})
                WITH p
                MATCH (s:Step {procedure_name: $proc_name, step_number: $step_num})
                MERGE (s)-[r:USES]->(p)
                RETURN p, r, s
                """
                tx.run(param_query, param=param, proc_name=procedure_name, step_num=step_number)
        
        # Create sequence relationship with previous step
        if step_number > 1:
            sequence_query = """
            MATCH (prev:Step {procedure_name: $proc_name, step_number: $prev_num})
            MATCH (curr:Step {procedure_name: $proc_name, step_number: $curr_num})
            MERGE (prev)-[r:FOLLOWED_BY]->(curr)
            RETURN prev, r, curr
            """
            tx.run(sequence_query, proc_name=procedure_name, 
                  prev_num=step_number-1, curr_num=step_number)

    def create_state_transition(self, from_state: str, to_state: str, 
                              trigger_event: str = None, conditions: List[str] = None,
                              metadata: Dict = None):
        """Create a state transition with optional trigger event and conditions."""
        with self.driver.session() as session:
            try:
                session.execute_write(self._create_state_transition, 
                                   from_state, to_state, trigger_event, conditions, metadata)
            except Exception as e:
                self.logger.error(f"Error creating transition {from_state}->{to_state}: {str(e)}")

    @staticmethod
    def _create_state_transition(tx, from_state: str, to_state: str, 
                               trigger_event: str = None, conditions: List[str] = None,
                               metadata: Dict = None):
        # Create the basic transition
        transition_query = """
        MATCH (s1:State {name: $from_state})
        MATCH (s2:State {name: $to_state})
        MERGE (s1)-[r:TRANSITIONS_TO]->(s2)
        SET r.conditions = $conditions,
            r.metadata = $metadata,
            r.timestamp = datetime()
        RETURN s1, r, s2
        """
        tx.run(transition_query, from_state=from_state, to_state=to_state,
               conditions=conditions, metadata=metadata)
        
        # If there's a trigger event, create it and link it
        if trigger_event:
            event_query = """
            MATCH (s1:State {name: $from_state})
            MATCH (s2:State {name: $to_state})
            MERGE (e:Event {name: $event})
            MERGE (s1)-[r1:TRIGGERED_BY]->(e)
            MERGE (e)-[r2:LEADS_TO]->(s2)
            RETURN s1, e, s2
            """
            tx.run(event_query, from_state=from_state, to_state=to_state, event=trigger_event)

    def clear_database(self):
        """Clear all nodes and relationships from the database."""
        with self.driver.session() as session:
            session.execute_write(self._clear_database)

    @staticmethod
    def _clear_database(tx):
        query = "MATCH (n) DETACH DELETE n"
        tx.run(query)

    def create_procedure_flow(self, procedure_name: str, steps: List[Dict]):
        """Create a complete procedure flow with steps, actions, and conditions."""
        with self.driver.session() as session:
            try:
                session.execute_write(self._create_procedure_flow, procedure_name, steps)
            except Exception as e:
                self.logger.error(f"Error creating procedure flow {procedure_name}: {str(e)}")

    def get_procedure_flow(self, procedure_name: str) -> Dict:
        """Get the complete execution flow of a procedure."""
        with self.driver.session() as session:
            try:
                return session.execute_read(self._get_procedure_flow, procedure_name)
            except Exception as e:
                self.logger.error(f"Error getting procedure flow {procedure_name}: {str(e)}")
                return None

    @staticmethod
    def _create_procedure_flow(tx, procedure_name: str, steps: List[Dict]):
        # Create procedure node
        proc_query = """
        MERGE (p:Procedure {name: $name})
        SET p.updated_at = datetime()
        RETURN p
        """
        tx.run(proc_query, name=procedure_name)

        # Process each step
        for i, step in enumerate(steps, 1):
            # Create step node with sequence
            step_query = """
            MERGE (s:Step {
                procedure_name: $proc_name,
                step_number: $step_num
            })
            SET s.description = $description,
                s.actor = $actor,
                s.action = $action,
                s.updated_at = datetime()
            WITH s
            MATCH (p:Procedure {name: $proc_name})
            MERGE (p)-[r:HAS_STEP]->(s)
            RETURN s
            """
            tx.run(step_query,
                  proc_name=procedure_name,
                  step_num=i,
                  description=step.get('description'),
                  actor=step.get('actor'),
                  action=step.get('action'))

            # Create action node and link to step
            if 'action' in step:
                action_query = """
                MERGE (a:Action {
                    name: $action,
                    actor: $actor
                })
                SET a.description = $description,
                    a.updated_at = datetime()
                WITH a
                MATCH (s:Step {
                    procedure_name: $proc_name,
                    step_number: $step_num
                })
                MERGE (s)-[r:PERFORMS]->(a)
                RETURN a
                """
                tx.run(action_query,
                      action=step['action'],
                      actor=step.get('actor'),
                      description=step.get('description'),
                      proc_name=procedure_name,
                      step_num=i)

            # Create parameters and link to action
            if 'parameters' in step:
                for param in step['parameters']:
                    param_query = """
                    MERGE (p:Parameter {name: $param})
                    WITH p
                    MATCH (s:Step {
                        procedure_name: $proc_name,
                        step_number: $step_num
                    })
                    MERGE (s)-[r:USES]->(p)
                    RETURN p
                    """
                    tx.run(param_query,
                          param=param,
                          proc_name=procedure_name,
                          step_num=i)

            # Create conditional nodes and decision paths
            if 'conditions' in step:
                for cond in step['conditions']:
                    cond_query = """
                    MERGE (c:Conditional {condition: $condition})
                    SET c.true_path = $true_path,
                        c.false_path = $false_path,
                        c.updated_at = datetime()
                    WITH c
                    MATCH (s:Step {
                        procedure_name: $proc_name,
                        step_number: $step_num
                    })
                    MERGE (s)-[r:HAS_CONDITION]->(c)
                    RETURN c
                    """
                    tx.run(cond_query,
                          condition=cond['condition'],
                          true_path=cond.get('true_path'),
                          false_path=cond.get('false_path'),
                          proc_name=procedure_name,
                          step_num=i)

            # Create sequence relationship with previous step
            if i > 1:
                sequence_query = """
                MATCH (prev:Step {
                    procedure_name: $proc_name,
                    step_number: $prev_num
                })
                MATCH (curr:Step {
                    procedure_name: $proc_name,
                    step_number: $curr_num
                })
                MERGE (prev)-[r:FOLLOWED_BY]->(curr)
                RETURN prev, curr
                """
                tx.run(sequence_query,
                      proc_name=procedure_name,
                      prev_num=i-1,
                      curr_num=i)

    @staticmethod
    def _get_procedure_flow(tx, procedure_name: str) -> Dict:
        # Query to get complete procedure flow
        query = """
        MATCH (p:Procedure {name: $name})
        OPTIONAL MATCH (p)-[:HAS_STEP]->(s:Step)
        OPTIONAL MATCH (s)-[:PERFORMS]->(a:Action)
        OPTIONAL MATCH (s)-[:USES]->(param:Parameter)
        OPTIONAL MATCH (s)-[:HAS_CONDITION]->(c:Conditional)
        OPTIONAL MATCH (s)-[:FOLLOWED_BY]->(next:Step)
        RETURN p, s, a, param, c, next
        ORDER BY s.step_number
        """
        result = tx.run(query, name=procedure_name)
        records = result.records()
        
        # Process results into a structured format
        procedure_flow = {
            "name": procedure_name,
            "steps": []
        }
        
        current_step = None
        for record in records:
            step = record.get("s")
            if step and (not current_step or current_step["number"] != step["step_number"]):
                current_step = {
                    "number": step["step_number"],
                    "description": step["description"],
                    "actor": step["actor"],
                    "action": record.get("a"),
                    "parameters": [],
                    "conditions": [],
                    "next_step": record.get("next", {}).get("step_number")
                }
                procedure_flow["steps"].append(current_step)
            
            param = record.get("param")
            if param and param["name"] not in current_step["parameters"]:
                current_step["parameters"].append(param["name"])
            
            cond = record.get("c")
            if cond and cond not in current_step["conditions"]:
                current_step["conditions"].append({
                    "condition": cond["condition"],
                    "true_path": cond["true_path"],
                    "false_path": cond["false_path"]
                })
        
        return procedure_flow

    def create_entity(self, name: str, entity_type: str):
        """Create an entity node with its type."""
        with self.driver.session() as session:
            try:
                session.execute_write(self._create_entity, name, entity_type)
            except Exception as e:
                self.logger.error(f"Error creating entity {name}: {str(e)}")

    @staticmethod
    def _create_entity(tx, name: str, entity_type: str):
        query = """
        MERGE (e:`3GPP Entity` {name: $name})
        SET e.type = $type,
            e.updated_at = datetime()
        RETURN e
        """
        tx.run(query, name=name, type=entity_type)

    def create_procedure(self, name: str, steps: List[Dict]):
        """Create a procedure with its steps."""
        with self.driver.session() as session:
            try:
                session.execute_write(self._create_procedure, name, steps)
            except Exception as e:
                self.logger.error(f"Error creating procedure {name}: {str(e)}")

    @staticmethod
    def _create_procedure(tx, name: str, steps: List[Dict]):
        # Create procedure node
        proc_query = """
        MERGE (p:Procedure {name: $name})
        SET p.updated_at = datetime()
        RETURN p
        """
        tx.run(proc_query, name=name)

        # Process each step
        for step in steps:
            step_query = """
            MERGE (s:Step {
                procedure_name: $proc_name,
                step_number: $step_num
            })
            SET s.description = $description,
                s.actor = $actor,
                s.action = $action,
                s.parameters = $parameters,
                s.updated_at = datetime()
            WITH s
            MATCH (p:Procedure {name: $proc_name})
            MERGE (p)-[r:HAS_STEP]->(s)
            RETURN s
            """
            tx.run(step_query,
                  proc_name=name,
                  step_num=step['step_number'],
                  description=step.get('description'),
                  actor=step.get('actor'),
                  action=step.get('action'),
                  parameters=step.get('parameters', []))

    def create_node(self, label: str, properties: Dict):
        """Create a node with the given label and properties."""
        with self.driver.session() as session:
            try:
                session.execute_write(self._create_node, label, properties)
            except Exception as e:
                self.logger.error(f"Error creating {label} node: {str(e)}")

    @staticmethod
    def _create_node(tx, label: str, properties: Dict):
        # Create dynamic query based on properties
        props_string = ", ".join(f"{k}: ${k}" for k in properties.keys())
        query = f"""
        MERGE (n:{label} {{{props_string}}})
        SET n.updated_at = datetime()
        RETURN n
        """
        tx.run(query, **properties)

    def create_relationship(self, from_label: str, from_key: str, 
                          rel_type: str, to_label: str, to_key: str):
        """Create a relationship between two nodes."""
        with self.driver.session() as session:
            try:
                session.execute_write(self._create_relationship, 
                                   from_label, from_key, rel_type, to_label, to_key)
            except Exception as e:
                self.logger.error(f"Error creating relationship: {str(e)}")

    @staticmethod
    def _create_relationship(tx, from_label: str, from_key: str, 
                           rel_type: str, to_label: str, to_key: str):
        query = f"""
        MATCH (a:{from_label} {{name: $from_key}})
        MATCH (b:{to_label} {{name: $to_key}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r.updated_at = datetime()
        RETURN a, r, b
        """
        tx.run(query, from_key=from_key, to_key=to_key)

def clean_text(text: str) -> str:
    """Clean text by removing unwanted characters and normalizing whitespace."""
    if not isinstance(text, str):
        return ""
    
    # Replace escaped newlines with spaces
    text = text.replace('\\n', ' ')
    
    # Replace other common escape sequences
    text = text.replace('\\t', ' ')
    text = text.replace('\\r', ' ')
    
    # Remove actual newlines and tabs
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('\r', ' ')
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove any remaining escape characters
    text = re.sub(r'\\[a-zA-Z]', ' ', text)
    
    return text

def normalize_entity_name(name: str) -> str:
    """Normalize entity name for consistent comparison."""
    # Clean the text first
    name = clean_text(name)
    
    # Convert to lowercase
    name = name.lower()
    
    # Remove any remaining special characters
    name = re.sub(r'[^a-z0-9\s]', '', name)
    
    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name)
    
    return name.strip()

def store_in_neo4j(processed_docs):
    """Store extracted entities and relationships in Neo4j with improved deduplication."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    driver = GraphDatabase.driver(uri, auth=(username, password))

    with driver.session() as session:
        # First, create a unique constraint if it doesn't exist
        try:
            session.run("""
                CREATE CONSTRAINT unique_entity_name IF NOT EXISTS
                FOR (e:`3GPP Entity`)
                REQUIRE (e.name, e.type) IS UNIQUE
            """)
        except Exception as e:
            print(f"Warning: Could not create constraint: {str(e)}")

        for doc in processed_docs:
            entities = doc["entities"]
            relationships = doc["relationships"]

            # Create nodes for each entity using MERGE to avoid duplicates
            for entity, entity_type in entities:
                try:
                    # Clean and normalize entity name
                    clean_name = clean_text(entity)
                    normalized_name = normalize_entity_name(entity)
                    clean_type = clean_text(entity_type)
                    
                    session.run(
                        """
                        MERGE (e:`3GPP Entity` {normalized_name: $normalized_name})
                        ON CREATE SET 
                            e.name = $name,
                            e.type = $type,
                            e.clean_name = $clean_name
                        ON MATCH SET 
                            e.name = CASE 
                                WHEN length($clean_name) > length(e.clean_name) THEN $name 
                                ELSE e.name 
                            END,
                            e.clean_name = CASE 
                                WHEN length($clean_name) > length(e.clean_name) THEN $clean_name 
                                ELSE e.clean_name 
                            END
                        """,
                        normalized_name=normalized_name,
                        name=entity,
                        clean_name=clean_name,
                        type=clean_type
                    )
                except Exception as e:
                    print(f"Error creating entity '{entity}': {str(e)}")

            # Create relationships between entities with deduplication
            for entity1, entity2, relationship_type, properties in relationships:
                try:
                    # Clean and normalize entity names and properties
                    norm_entity1 = normalize_entity_name(entity1)
                    norm_entity2 = normalize_entity_name(entity2)
                    clean_rel_type = clean_text(relationship_type)
                    
                    # Clean all property values
                    clean_properties = {}
                    for key, value in properties.items():
                        if isinstance(value, str):
                            clean_properties[key] = clean_text(value)
                        else:
                            clean_properties[key] = value
                    
                    session.run(
                        """
                        MATCH (e1:`3GPP Entity` {normalized_name: $entity1}), 
                              (e2:`3GPP Entity` {normalized_name: $entity2})
                        WHERE NOT EXISTS((e1)-[r:$rel_type]->(e2) WHERE r.description = $desc)
                        MERGE (e1)-[r:$rel_type]->(e2)
                        ON CREATE SET r += $props
                        """,
                        entity1=norm_entity1,
                        entity2=norm_entity2,
                        rel_type=clean_rel_type,
                        desc=clean_properties.get('description', ''),
                        props=clean_properties
                    )
                except Exception as e:
                    print(f"Error creating relationship between '{entity1}' and '{entity2}': {str(e)}")

    driver.close()