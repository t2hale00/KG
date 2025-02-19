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

# Configure logging to be less verbose
logging.getLogger("neo4j").setLevel(logging.WARNING)  # Only show WARNING and above for neo4j
logger = logging.getLogger(__name__)

# Neo4j Configuration (from environment variables)
URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

# Cache file for Neo4j state
NEO4J_CACHE_FILE = "neo4j_cache.json"

class KnowledgeGraph:
    def __init__(self, uri: str, user: str, password: str):
        """Initialize Neo4j connection and schema if needed."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.logger = logging.getLogger(__name__)
        self._ensure_schema()

    def close(self):
        """Close the Neo4j driver."""
        self.driver.close()

    def _ensure_schema(self):
        """Ensure Neo4j schema exists with necessary constraints and indexes."""
        with self.driver.session() as session:
            try:
                # Create constraints
                constraints = [
                    # State constraints
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (s:State) REQUIRE s.name IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (s:State) REQUIRE s.id IS UNIQUE",
                    
                    # Event constraints
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Event) REQUIRE e.name IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE",
                    
                    # Network element constraints
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:NetworkElement) REQUIRE n.name IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:NetworkElement) REQUIRE n.id IS UNIQUE",
                    
                    # Conditional constraints
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Conditional) REQUIRE c.name IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Conditional) REQUIRE c.id IS UNIQUE",
                    
                    # Parameter constraints
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Parameter) REQUIRE p.name IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Parameter) REQUIRE p.id IS UNIQUE"
                ]
                
                # Create indexes
                indexes = [
                    # State indexes
                    "CREATE INDEX IF NOT EXISTS FOR (s:State) ON (s.state_type)",
                    "CREATE INDEX IF NOT EXISTS FOR (s:State) ON (s.entry_conditions)",
                    "CREATE INDEX IF NOT EXISTS FOR (s:State) ON (s.exit_conditions)",
                    
                    # Event indexes
                    "CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.event_type)",
                    "CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.source_entity)",
                    "CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.target_entity)",
                    
                    # Network element indexes
                    "CREATE INDEX IF NOT EXISTS FOR (n:NetworkElement) ON (n.element_type)",
                    "CREATE INDEX IF NOT EXISTS FOR (n:NetworkElement) ON (n.role)",
                    
                    # Conditional indexes
                    "CREATE INDEX IF NOT EXISTS FOR (c:Conditional) ON (c.condition_type)",
                    "CREATE INDEX IF NOT EXISTS FOR (c:Conditional) ON (c.true_path)",
                    "CREATE INDEX IF NOT EXISTS FOR (c:Conditional) ON (c.false_path)",
                    
                    # Parameter indexes
                    "CREATE INDEX IF NOT EXISTS FOR (p:Parameter) ON (p.parameter_type)",
                    "CREATE INDEX IF NOT EXISTS FOR (p:Parameter) ON (p.format)",
                    "CREATE INDEX IF NOT EXISTS FOR (p:Parameter) ON (p.mandatory)"
                ]
                
                # Execute all schema creation queries
                for query in constraints + indexes:
                    session.run(query)
                
                self.logger.info("✓ Schema initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Error initializing schema: {str(e)}")

    def initialize_schema(self):
        return

    def create_state(self, name: str, state_type: str, description: str = None, conditions: List[str] = None):
        """Create a state node with its properties."""
        with self.driver.session() as session:
            try:
                session.execute_write(self._create_state, name, state_type, description, conditions)
            except Exception as e:
                self.logger.error(f"Error creating state {name}: {str(e)}")

    def create_event(self, name: str, event_type: str, source_entity: str, target_entity: str, parameters: List[str], metadata: Dict[str, Any]) -> None:
        """Create an event node in Neo4j.
        
        Args:
            name: Name of the event
            event_type: Type of event (message, timer, internal)
            source_entity: Entity that generates the event
            target_entity: Entity that receives the event
            parameters: List of parameters associated with the event
            metadata: Additional metadata (protocol, message_type, timer_value, retry_count)
        """
        query = """
        MERGE (e:Event {name: $name})
        SET e.event_type = $event_type,
            e.source_entity = $source_entity,
            e.target_entity = $target_entity,
            e.parameters = $parameters,
            e.protocol = $protocol,
            e.message_type = $message_type,
            e.timer_value = $timer_value,
            e.retry_count = $retry_count,
            e.updated_at = datetime()
        RETURN e
        """
        
        with self.driver.session() as session:
            try:
                session.run(
                    query,
                    name=name,
                    event_type=event_type,
                    source_entity=source_entity,
                    target_entity=target_entity,
                    parameters=parameters,
                    protocol=metadata.get("protocol"),
                    message_type=metadata.get("message_type"),
                    timer_value=metadata.get("timer_value"),
                    retry_count=metadata.get("retry_count")
                )
                self.logger.info(f"Created/updated Event node: {name}")
            except Exception as e:
                self.logger.error(f"Error creating Event node {name}: {str(e)}")
                raise

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
        """Create a state node in Neo4j."""
        query = """
        MERGE (s:State {name: $name})
        SET s.state_type = $state_type,
            s.description = $description,
            s.entry_conditions = $conditions,
            s.updated_at = datetime()
        RETURN s
        """
        result = tx.run(query, name=name, state_type=state_type, 
                       description=description, conditions=conditions)
        return result.single()

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
            MATCH (prev:Step {
                procedure_name: $proc_name,
                step_number: $prev_num
            })
            MATCH (curr:Step {
                procedure_name: $proc_name,
                step_number: $curr_num
            })
            MERGE (prev)-[r:FOLLOWED_BY]->(curr)
            RETURN prev, r, curr
            """
            tx.run(sequence_query,
                  proc_name=procedure_name,
                  prev_num=step_number-1,
                  curr_num=step_number)

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
        """Create a state transition in Neo4j."""
        # Create the base transition
        query = """
        MATCH (s1:State {name: $from_state})
        MATCH (s2:State {name: $to_state})
        MERGE (s1)-[r:TRANSITIONS_TO]->(s2)
        SET r.conditions = $conditions,
            r.metadata = $metadata,
            r.updated_at = datetime()
        """
        tx.run(query, from_state=from_state, to_state=to_state,
               conditions=conditions, metadata=metadata)
        
        # If there's a trigger event, create it and link it
        if trigger_event:
            query = """
            MATCH (s1:State {name: $from_state})
            MATCH (s2:State {name: $to_state})
            MERGE (e:Event {name: $trigger_event})
            SET e.event_type = 'transition_trigger',
                e.updated_at = datetime()
            MERGE (e)-[t:TRIGGERS]->(s2)
            SET t.from_state = $from_state,
                t.updated_at = datetime()
            """
            tx.run(query, from_state=from_state, to_state=to_state,
                   trigger_event=trigger_event)

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
        MERGE (e:ThreeGPPEntity {name: $name})
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
        MATCH (a:ThreeGPPEntity {{name: $from_key}})
        MATCH (b:ThreeGPPEntity {{name: $to_key}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r.updated_at = datetime()
        RETURN a, r, b
        """
        tx.run(query, from_key=from_key, to_key=to_key)

    def store_3gpp_entities(self, data: Dict[str, Any]) -> None:
        """Store 3GPP entities in Neo4j."""
        with self.driver.session() as session:
            try:
                # Create nodes
                for node in data.get("nodes", []):
                    node_id = node.get("id")
                    node_type = node.get("type")
                    node_name = node.get("name")
                    properties = node.get("properties", {})

                    if not node_id or not node_type or not node_name:
                        continue

                    # Base query for all node types
                    base_query = f"""
                    MERGE (n:{node_type} {{name: $name}})
                    SET n.id = $id,
                        n.description = $description,
                        n.updated_at = datetime()
                    """

                    # Additional properties based on node type
                    if node_type == "State":
                        query = base_query + """
                        SET n.state_type = $state_type,
                            n.entry_conditions = $entry_conditions,
                            n.exit_conditions = $exit_conditions,
                            n.metadata = $metadata
                        """
                        session.run(query,
                                  name=node_name,
                                  id=node_id,
                                  description=properties.get("description", ""),
                                  state_type=properties.get("state_type"),
                                  entry_conditions=properties.get("entry_conditions", []),
                                  exit_conditions=properties.get("exit_conditions", []),
                                  metadata=properties.get("metadata", {}))

                    elif node_type == "Event":
                        query = base_query + """
                        SET n.event_type = $event_type,
                            n.source_entity = $source_entity,
                            n.target_entity = $target_entity,
                            n.parameters = $parameters,
                            n.metadata = $metadata
                        """
                        session.run(query,
                                  name=node_name,
                                  id=node_id,
                                  description=properties.get("description", ""),
                                  event_type=properties.get("event_type"),
                                  source_entity=properties.get("source_entity"),
                                  target_entity=properties.get("target_entity"),
                                  parameters=properties.get("parameters", []),
                                  metadata=properties.get("metadata", {}))

                    elif node_type == "NetworkElement":
                        query = base_query + """
                        SET n.element_type = $element_type,
                            n.role = $role,
                            n.interfaces = $interfaces,
                            n.metadata = $metadata
                        """
                        session.run(query,
                                  name=node_name,
                                  id=node_id,
                                  description=properties.get("description", ""),
                                  element_type=properties.get("element_type"),
                                  role=properties.get("role"),
                                  interfaces=properties.get("interfaces", []),
                                  metadata=properties.get("metadata", {}))

                    elif node_type == "Conditional":
                        query = base_query + """
                        SET n.condition_type = $condition_type,
                            n.true_path = $true_path,
                            n.false_path = $false_path,
                            n.parameters = $parameters,
                            n.metadata = $metadata
                        """
                        session.run(query,
                                  name=node_name,
                                  id=node_id,
                                  description=properties.get("description", ""),
                                  condition_type=properties.get("condition_type"),
                                  true_path=properties.get("true_path"),
                                  false_path=properties.get("false_path"),
                                  parameters=properties.get("parameters", []),
                                  metadata=properties.get("metadata", {}))

                    elif node_type == "Parameter":
                        query = base_query + """
                        SET n.parameter_type = $parameter_type,
                            n.format = $format,
                            n.mandatory = $mandatory,
                            n.metadata = $metadata
                        """
                        session.run(query,
                                  name=node_name,
                                  id=node_id,
                                  description=properties.get("description", ""),
                                  parameter_type=properties.get("parameter_type"),
                                  format=properties.get("format"),
                                  mandatory=properties.get("mandatory", False),
                                  metadata=properties.get("metadata", {}))

                # Create edges
                for edge in data.get("edges", []):
                    source = edge.get("source")
                    target = edge.get("target")
                    edge_type = edge.get("type")
                    properties = edge.get("properties", {})

                    if not source or not target or not edge_type:
                        continue

                    # Create relationship based on edge type
                    query = """
                    MATCH (source {name: $source_name})
                    MATCH (target {name: $target_name})
                    MERGE (source)-[r:$edge_type]->(target)
                    SET r += $properties,
                        r.updated_at = datetime()
                    """
                    session.run(query,
                              source_name=source,
                              target_name=target,
                              edge_type=edge_type.upper(),
                              properties=properties)

            except Exception as e:
                self.logger.error(f"Error storing 3GPP entities: {str(e)}")
                raise

    def _get_protocol_description(self, protocol_name: str) -> str:
        """Get the description for a protocol."""
        protocol_descriptions = {
            'NAS': 'Non-Access Stratum protocol for communication between UE and core network',
            'RRC': 'Radio Resource Control protocol for radio resource management',
            'GTP': 'GPRS Tunneling Protocol for user and control plane tunneling',
            'NGAP': 'NG Application Protocol for control plane signaling',
            'S1AP': 'S1 Application Protocol for communication between eNB and MME',
            'X2AP': 'X2 Application Protocol for communication between eNBs'
        }
        return protocol_descriptions.get(protocol_name, "Protocol used in 3GPP communications")

    def _get_network_element_description(self, element_name: str) -> str:
        """Get the description for a network element."""
        element_descriptions = {
            'UE': 'User Equipment - The end-user device in the network',
            'eNB': 'evolved NodeB - The base station in LTE networks',
            'gNB': 'next generation NodeB - The base station in 5G networks',
            'AMF': 'Access and Mobility Management Function - Handles access control and mobility',
            'SMF': 'Session Management Function - Manages user sessions and connectivity',
            'UPF': 'User Plane Function - Handles user data forwarding and routing',
            'AUSF': 'Authentication Server Function - Handles user authentication',
            'UDM': 'Unified Data Management - Manages user subscription data',
            'PCF': 'Policy Control Function - Manages network policies',
            'NRF': 'Network Repository Function - Service discovery and registration',
            'NSSF': 'Network Slice Selection Function - Handles network slice selection',
            'SGW': 'Serving Gateway - Routes and forwards user data packets',
            'PGW': 'PDN Gateway - Provides connectivity to external networks',
            'MME': 'Mobility Management Entity - Controls mobility in LTE networks'
        }
        return element_descriptions.get(element_name, "Network element in 3GPP architecture")

    @staticmethod
    def _store_3gpp_relationships(tx, relationships: List[Dict[str, Any]], source_doc: str):
        for rel in relationships:
            # Extract common properties
            base_properties = {
                'context': rel.get('context'),
                'source_doc': source_doc,
                'updated_at': 'datetime()',
                'conditions': rel.get('conditions', []),
                'triggers': rel.get('triggers', [])
            }

            # Add any custom properties from the relationship
            if 'properties' in rel:
                base_properties.update(rel['properties'])

            if rel['type'] == 'sends':
                query = """
                MATCH (source:ThreeGPPEntity {name: $source_name})
                MATCH (target:ThreeGPPEntity {name: $target_name})
                MERGE (source)-[r:SENDS]->(target)
                SET r = $properties,
                    r.message = $message
                WITH source, r, target
                FOREACH (condition IN $conditions |
                    MERGE (c:Conditional {condition: condition})
                    MERGE (r)-[:HAS_CONDITION]->(c)
                )
                FOREACH (trigger IN $triggers |
                    MERGE (e:Event {name: trigger})
                    MERGE (r)-[:TRIGGERED_BY]->(e)
                )
                """
                tx.run(query, 
                      source_name=rel['source'],
                      target_name=rel['target'],
                      message=rel.get('message'),
                      properties=base_properties,
                      conditions=rel.get('conditions', []),
                      triggers=rel.get('triggers', []))

            elif rel['type'] == 'authenticates':
                query = """
                MATCH (source:ThreeGPPEntity {name: $source_name})
                MATCH (target:ThreeGPPEntity {name: $target_name})
                MERGE (source)-[r:AUTHENTICATES]->(target)
                SET r = $properties
                WITH source, r, target
                FOREACH (condition IN $conditions |
                    MERGE (c:Conditional {condition: condition})
                    MERGE (r)-[:HAS_CONDITION]->(c)
                )
                FOREACH (trigger IN $triggers |
                    MERGE (e:Event {name: trigger})
                    MERGE (r)-[:TRIGGERED_BY]->(e)
                )
                """
                tx.run(query,
                      source_name=rel['source'],
                      target_name=rel['target'],
                      properties=base_properties,
                      conditions=rel.get('conditions', []),
                      triggers=rel.get('triggers', []))

            elif rel['type'] == 'establishes':
                query = """
                MATCH (source:ThreeGPPEntity {name: $source_name})
                MATCH (target:ThreeGPPEntity {name: $target_name})
                MERGE (source)-[r:ESTABLISHES]->(target)
                SET r = $properties,
                    r.connection_type = $connection_type
                WITH source, r, target
                FOREACH (condition IN $conditions |
                    MERGE (c:Conditional {condition: condition})
                    MERGE (r)-[:HAS_CONDITION]->(c)
                )
                FOREACH (trigger IN $triggers |
                    MERGE (e:Event {name: trigger})
                    MERGE (r)-[:TRIGGERED_BY]->(e)
                )
                """
                tx.run(query,
                      source_name=rel['source'],
                      target_name=rel['target'],
                      connection_type=rel.get('message'),
                      properties=base_properties,
                      conditions=rel.get('conditions', []),
                      triggers=rel.get('triggers', []))

            elif rel['type'] == 'state_transition':
                query = """
                MATCH (from_state:ThreeGPPEntity {name: $from_state})
                MATCH (to_state:ThreeGPPEntity {name: $to_state})
                MERGE (from_state)-[r:TRANSITIONS_TO]->(to_state)
                SET r = $properties
                WITH from_state, r, to_state
                FOREACH (condition IN $conditions |
                    MERGE (c:Conditional {condition: condition})
                    MERGE (r)-[:HAS_CONDITION]->(c)
                )
                FOREACH (trigger IN $triggers |
                    MERGE (e:Event {name: trigger, type: 'transition_trigger'})
                    SET e.description = 'Event triggering state transition'
                    MERGE (r)-[:TRIGGERED_BY]->(e)
                )
                """
                tx.run(query,
                      from_state=rel['source'],
                      to_state=rel['target'],
                      properties=base_properties,
                      conditions=rel.get('conditions', []),
                      triggers=rel.get('triggers', []))

    def store_events_and_triggers(self, events: List[Dict[str, Any]], source_doc: str):
        """Store events and their triggers in Neo4j."""
        with self.driver.session() as session:
            try:
                session.execute_write(self._store_events_and_triggers, events, source_doc)
            except Exception as e:
                self.logger.error(f"Error storing events and triggers: {str(e)}")

    @staticmethod
    def _store_events_and_triggers(tx, events: List[Dict[str, Any]], source_doc: str):
        for event in events:
            query = """
            MERGE (e:Event {name: $name})
            SET e.type = $type,
                e.description = $description,
                e.source_doc = $source_doc,
                e.updated_at = datetime()
            WITH e
            FOREACH (trigger IN $triggers |
                MERGE (t:Trigger {name: trigger})
                MERGE (t)-[:TRIGGERS]->(e)
            )
            FOREACH (condition IN $conditions |
                MERGE (c:Conditional {condition: condition})
                MERGE (e)-[:HAS_CONDITION]->(c)
            )
            """
            tx.run(query,
                  name=event['name'],
                  type=event.get('type', 'protocol_event'),
                  description=event.get('description', 'Event in 3GPP protocol'),
                  source_doc=source_doc,
                  triggers=event.get('triggers', []),
                  conditions=event.get('conditions', []))

    def store_conditions(self, conditions: List[Dict[str, Any]], source_doc: str):
        """Store conditions and their relationships in Neo4j."""
        with self.driver.session() as session:
            try:
                session.execute_write(self._store_conditions, conditions, source_doc)
            except Exception as e:
                self.logger.error(f"Error storing conditions: {str(e)}")

    @staticmethod
    def _store_conditions(tx, conditions: List[Dict[str, Any]], source_doc: str):
        """Store conditions in Neo4j."""
        for condition in conditions:
            # Create condition node
            query = (
                "CREATE (c:Condition {condition: $condition, source_doc: $source_doc})"
            )
            tx.run(query, condition=condition, source_doc=source_doc)

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
    """Store extracted entities and relationships in Neo4j with improved state machine handling."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    driver = GraphDatabase.driver(uri, auth=(username, password))

    def flatten_properties(props):
        """Flatten nested properties and convert to primitive types."""
        flattened = {}
        for key, value in props.items():
            if isinstance(value, (str, int, float, bool)):
                flattened[key] = value
            elif isinstance(value, list):
                # Convert list items to strings if they're not primitive types
                flattened[key] = [str(item) if not isinstance(item, (str, int, float, bool)) else item 
                                for item in value]
            elif isinstance(value, dict):
                # For metadata dictionaries, keep them as separate properties
                if key == "metadata":
                    for meta_key, meta_value in value.items():
                        flattened[f"metadata_{meta_key}"] = str(meta_value)
                else:
                    # For other dictionaries, convert to string
                    flattened[key + "_str"] = str(value)
            else:
                # Convert any other types to string
                flattened[key] = str(value)
        return flattened

    with driver.session() as session:
        # First, clear all existing nodes and relationships
        print("Clearing existing database...")
        try:
            session.run("MATCH (n) DETACH DELETE n")
            print("✓ Database cleared successfully")
        except Exception as e:
            print(f"Warning: Could not clear database: {str(e)}")

        # Process each document
        for doc in processed_docs:
            entities = doc.get("nodes", [])
            relationships = doc.get("edges", [])

            # Create nodes for each entity
            print(f"Creating {len(entities)} nodes...")
            for entity in entities:
                try:
                    # Determine the correct label based on node type
                    node_type = entity["type"]
                    if node_type == "state":
                        label = "State"
                    elif node_type == "event":
                        label = "Event"
                    elif node_type == "network_element":
                        label = "NetworkElement"
                    elif node_type == "conditional":
                        label = "Conditional"
                    else:
                        label = "Node"
                    
                    # Clean and normalize entity name
                    clean_name = clean_text(entity["name"])
                    normalized_name = normalize_entity_name(entity["name"])
                    
                    # Flatten properties to primitive types
                    properties = flatten_properties(entity.get("properties", {}))
                    
                    # Create node with appropriate label and properties
                    query = f"""
                    MERGE (n:{label} {{name: $name}})
                    SET n.normalized_name = $normalized_name,
                        n.clean_name = $clean_name,
                        n += $properties
                    """
                    session.run(
                        query,
                        name=entity["name"],
                        normalized_name=normalized_name,
                        clean_name=clean_name,
                        properties=properties
                    )
                except Exception as e:
                    print(f"Error creating entity '{entity['name']}': {str(e)}")

            # Create relationships between nodes
            print(f"Creating {len(relationships)} relationships...")
            for edge in relationships:
                try:
                    # Flatten relationship properties
                    props = flatten_properties(edge.get("properties", {}))
                    
                    # Create relationship with explicit type
                    query = f"""
                    MATCH (source) WHERE source.name = $source_name
                    MATCH (target) WHERE target.name = $target_name
                    MERGE (source)-[r:{edge['type'].upper()}]->(target)
                    SET r += $props
                    """
                    
                    session.run(
                        query,
                        source_name=edge["source"],
                        target_name=edge["target"],
                        props=props
                    )
                except Exception as e:
                    print(f"Error creating relationship between '{edge['source']}' and '{edge['target']}': {str(e)}")

        print("✓ Data loading completed successfully")

    driver.close()

def process_3gpp_data(json_file: str):
    """Process extracted 3GPP data and store in Neo4j."""
    # Read Neo4j credentials from environment
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not password:
        raise ValueError("NEO4J_PASSWORD environment variable not set")
    
    # Initialize Neo4j connection
    graph = KnowledgeGraph(uri, user, password)
    
    try:
        # Initialize schema
        graph.initialize_schema()
        
        # Read the JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process each procedure
        for procedure in data['procedures']:
            # Store entities
            graph.store_3gpp_entities(procedure['entities'], data['source_document'])
            
            # Store relationships
            if 'relationships' in procedure:
                graph.store_3gpp_relationships(procedure['relationships'], data['source_document'])
        
        print(f"Successfully processed {len(data['procedures'])} procedures")
        
    except Exception as e:
        print(f"Error processing 3GPP data: {str(e)}")
    finally:
        graph.close()

def load_graph_data(json_file: str):
    """Load extracted graph data into Neo4j."""
    # Read Neo4j credentials from environment
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not password:
        raise ValueError("NEO4J_PASSWORD environment variable not set")
    
    # Initialize Neo4j connection
    graph = KnowledgeGraph(uri, user, password)
    
    try:
        # Initialize schema
        graph._ensure_schema()
        
        # Read the JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Store entities and relationships
        graph.store_3gpp_entities(data)
        
        print(f"Successfully loaded graph data from {json_file}")
        print(f"Nodes created: {len(data.get('nodes', []))}")
        print(f"Edges created: {len(data.get('edges', []))}")
        
        # Print node type distribution
        node_types = {}
        for node in data.get("nodes", []):
            node_type = node.get("type")
            if node_type:
                node_types[node_type] = node_types.get(node_type, 0) + 1
        
        print("\nNode type distribution:")
        for ntype, count in sorted(node_types.items()):
            print(f"- {ntype}: {count}")
        
        # Print edge type distribution
        edge_types = {}
        for edge in data.get("edges", []):
            edge_type = edge.get("type")
            if edge_type:
                edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        print("\nEdge type distribution:")
        for etype, count in sorted(edge_types.items()):
            print(f"- {etype}: {count}")
        
    except Exception as e:
        print(f"Error loading graph data: {str(e)}")
        raise
    finally:
        graph.close()

def verify_neo4j_data():
    """Verify the data loaded in Neo4j and print statistics."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not password:
        raise ValueError("NEO4J_PASSWORD environment variable not set")
    
    driver = GraphDatabase.driver(uri, auth=(username, password))

    with driver.session() as session:
        # Count nodes by label
        result = session.run("""
            CALL db.labels() YIELD label
            CALL {
                WITH label
                MATCH (n)
                WHERE label in labels(n)
                RETURN count(n) as count
            }
            RETURN label, count
            ORDER BY count DESC
        """)
        print("\nNodes by label:")
        for record in result:
            print(f"  {record['label']}: {record['count']}")
        
        # Count relationships by type
        result = session.run("""
            CALL db.relationshipTypes() YIELD relationshipType
            CALL {
                WITH relationshipType
                MATCH ()-[r]->()
                WHERE type(r) = relationshipType
                RETURN count(r) as count
            }
            RETURN relationshipType, count
            ORDER BY count DESC
        """)
        print("\nRelationships by type:")
        for record in result:
            print(f"  {record['relationshipType']}: {record['count']}")
        
        # Verify state machine structure
        result = session.run("""
            MATCH (s:State)
            WITH s.state_type as type, count(*) as count
            RETURN type, count
            ORDER BY count DESC
        """)
        print("\nStates by type:")
        for record in result:
            print(f"  {record['type']}: {record['count']}")
        
        # Verify event triggers
        result = session.run("""
            MATCH (e:Event)-[:TRIGGERS]->(s:State)
            RETURN e.event_type as type, count(*) as count
            ORDER BY count DESC
        """)
        print("\nEvent triggers by type:")
        for record in result:
            print(f"  {record['type']}: {record['count']}")
    
    driver.close()

def print_visualization_queries():
    """Print example queries for visualizing the state machine in Neo4j Browser."""
    print("\nExample queries for Neo4j Browser visualization:")
    
    print("\n1. View complete state machine flow:")
    print("""
    MATCH (s:State)
    OPTIONAL MATCH (s)-[r]->(t)
    RETURN s, r, t;
    """)
    
    print("\n2. View state transitions with events:")
    print("""
    MATCH (s1:State)-[r:TRANSITIONS_TO]->(s2:State)
    OPTIONAL MATCH (e:Event)-[tr:TRIGGERS]->(s2)
    RETURN s1, r, s2, e, tr;
    """)
    
    print("\n3. View conditional transitions:")
    print("""
    MATCH (s:State)-[r]->(c:Conditional)
    OPTIONAL MATCH (c)-[t:TRANSITIONS_TO]->(s2:State)
    RETURN s, r, c, t, s2;
    """)
    
    print("\n4. View network element interactions:")
    print("""
    MATCH (n:NetworkElement)-[r]->(m:NetworkElement)
    OPTIONAL MATCH (e:Event)-[tr:TRIGGERS]->(s:State)
    WHERE e.metadata_source = n.name AND e.metadata_target = m.name
    RETURN n, r, m, e, tr, s;
    """)
    
    print("\n5. View complete attach procedure flow:")
    print("""
    MATCH p = (start:State {name: 'UE_POWERED_ON'})-[*]->(end:State {name: 'UE_ATTACHED'})
    UNWIND relationships(p) as r
    WITH DISTINCT r
    MATCH (s)-[r]->(t)
    RETURN s, r, t;
    """)

if __name__ == "__main__":
    # Load the extracted graph data
    json_file = "processed_data/3gpp_graph.json"
    if os.path.exists(json_file):
        load_graph_data(json_file)
        verify_neo4j_data()
        print_visualization_queries()
    else:
        print(f"JSON file not found: {json_file}")