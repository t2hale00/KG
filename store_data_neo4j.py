import os
import re
import json
from typing import List, Dict
from neo4j import GraphDatabase
import logging
from dotenv import load_dotenv
import time
import random

# Load environment variables
load_dotenv()

# Configure logging
logging.getLogger("neo4j").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class Neo4jConnector:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.driver = None
        self._connect()
        self._reset_and_setup_schema()
    
    def _connect(self):
        """Establish connection to Neo4j"""
        if not self.password:
            raise ValueError("Neo4j password not found in environment variables")
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("✓ Connected to Neo4j successfully")
        except Exception as e:
            print(f"Error connecting to Neo4j: {str(e)}")
            raise
    
    def _reset_and_setup_schema(self):
        """Reset database and set up optimized schema"""
        with self.driver.session() as session:
            try:
                # Delete all data
                print("Cleaning database...")
                session.run("MATCH (n) DETACH DELETE n")
                
                # Drop all constraints and indexes
                print("Dropping existing schema...")
                session.run("CALL apoc.schema.assert({},{},true)")
                
                # Create optimized constraints and indexes
                print("Creating new schema...")
                constraints = [
                    # Node uniqueness constraints
                    "CREATE CONSTRAINT node_id IF NOT EXISTS FOR (n:NetworkElement) REQUIRE n.id IS UNIQUE",
                    "CREATE CONSTRAINT node_name IF NOT EXISTS FOR (n:NetworkElement) REQUIRE n.name IS UNIQUE",
                    "CREATE CONSTRAINT state_id IF NOT EXISTS FOR (n:State) REQUIRE n.id IS UNIQUE",
                    "CREATE CONSTRAINT state_name IF NOT EXISTS FOR (n:State) REQUIRE n.name IS UNIQUE",
                    "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (n:Event) REQUIRE n.id IS UNIQUE",
                    "CREATE CONSTRAINT event_name IF NOT EXISTS FOR (n:Event) REQUIRE n.name IS UNIQUE"
                ]
                
                indexes = [
                    # Indexes for faster property lookups
                    "CREATE INDEX node_type IF NOT EXISTS FOR (n:NetworkElement) ON (n.type)",
                    "CREATE INDEX state_type IF NOT EXISTS FOR (n:State) ON (n.type)",
                    "CREATE INDEX event_type IF NOT EXISTS FOR (n:Event) ON (n.type)",
                    # Composite indexes for relationship queries
                    "CREATE INDEX rel_batch IF NOT EXISTS FOR ()-[r:SENDS|TRANSITIONS_TO|TRIGGERS]->() ON (r.batch_id)",
                    "CREATE INDEX node_batch IF NOT EXISTS FOR (n) ON (n.batch_id)"
                ]
                
                for constraint in constraints:
                    try:
                        session.run(constraint)
                    except Exception as e:
                        print(f"Warning: Could not create constraint: {str(e)}")
                
                for index in indexes:
                    try:
                        session.run(index)
                    except Exception as e:
                        print(f"Warning: Could not create index: {str(e)}")
                
                print("✓ Neo4j schema setup completed")
                
            except Exception as e:
                print(f"Error setting up schema: {str(e)}")
                raise
    
    def store_batch(self, nodes: List[Dict], edges: List[Dict], batch_id: str) -> bool:
        """Store a batch of nodes and edges with optimized bulk loading"""
        with self.driver.session() as session:
            try:
                # Use a single transaction for the entire batch
                with session.begin_transaction() as tx:
                    # Bulk create nodes using UNWIND
                    if nodes:
                        node_query = """
                        UNWIND $nodes as node
                        MERGE (n:`${node.type}` {id: node.id})
                        ON CREATE SET 
                            n.name = node.name,
                            n.properties = node.properties,
                            n.created_at = datetime(),
                            n.batch_id = $batch_id
                        ON MATCH SET 
                            n.properties = node.properties,
                            n.updated_at = datetime(),
                            n.last_batch_id = $batch_id
                        """
                        tx.run(node_query, {"nodes": nodes, "batch_id": batch_id})
                    
                    # Bulk create relationships using UNWIND
                    if edges:
                        edge_query = """
                        UNWIND $edges as edge
                        MATCH (source {id: edge.source})
                        MATCH (target {id: edge.target})
                        MERGE (source)-[r:`${edge.type}`]->(target)
                        ON CREATE SET 
                            r.properties = edge.properties,
                            r.created_at = datetime(),
                            r.batch_id = $batch_id
                        ON MATCH SET 
                            r.properties = edge.properties,
                            r.updated_at = datetime(),
                            r.last_batch_id = $batch_id
                        """
                        tx.run(edge_query, {"edges": edges, "batch_id": batch_id})
                    
                    tx.commit()
                    return True
                    
            except Exception as e:
                print(f"Error storing batch {batch_id}: {str(e)}")
                return False
    
    def verify_batch(self, batch_id: str) -> Dict[str, int]:
        """Verify the data loaded in a specific batch with optimized queries"""
        with self.driver.session() as session:
            try:
                # Use more efficient counting queries
                node_result = session.run("""
                    MATCH (n)
                    WHERE n.batch_id = $batch_id
                    RETURN labels(n)[0] as label, count(*) as count
                """, {'batch_id': batch_id})
                
                rel_result = session.run("""
                    MATCH ()-[r]->()
                    WHERE r.batch_id = $batch_id
                    RETURN type(r) as type, count(*) as count
                """, {'batch_id': batch_id})
                
                stats = {
                    'nodes': {record['label']: record['count'] for record in node_result},
                    'relationships': {record['type']: record['count'] for record in rel_result}
                }
                
                return stats
                
            except Exception as e:
                print(f"Error verifying batch {batch_id}: {str(e)}")
                return {'nodes': {}, 'relationships': {}}
    
    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()

# Singleton instance
_neo4j_connector = None

def get_neo4j_connector():
    """Get or create Neo4j connector instance"""
    global _neo4j_connector
    if _neo4j_connector is None:
        _neo4j_connector = Neo4jConnector()
    return _neo4j_connector

def store_batch_in_neo4j(nodes: List[Dict], edges: List[Dict], batch_id: str) -> bool:
    """Store a batch of nodes and edges in Neo4j with verification"""
    connector = get_neo4j_connector()
    success = connector.store_batch(nodes, edges, batch_id)
    
    if success:
        # Verify the stored data
        stats = connector.verify_batch(batch_id)
        print(f"\nBatch {batch_id} verification:")
        print("Nodes stored:")
        for label, count in stats['nodes'].items():
            print(f"  - {label}: {count}")
        print("Relationships stored:")
        for rel_type, count in stats['relationships'].items():
            print(f"  - {rel_type}: {count}")
    
    return success

def verify_neo4j_data():
    """Verify the data loaded in Neo4j and print statistics"""
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

def print_visualization_queries():
    """Print example queries for visualizing the graph in Neo4j Browser"""
    print("\nExample queries for Neo4j Browser visualization:")
    
    print("\n1. View network elements and their messages:")
    print("""
    MATCH (n:NetworkElement)-[r:SENDS]->(m:NetworkElement)
    RETURN n, r, m;
    """)
    
    print("\n2. View state machine:")
    print("""
    MATCH (s1:State)-[r:TRANSITIONS_TO]->(s2:State)
    RETURN s1, r, s2;
    """)
    
    print("\n3. View events and their related states:")
    print("""
    MATCH (e:Event)-[r]->(s:State)
    RETURN e, r, s;
    """)

def load_graph_data(json_file: str) -> bool:
    """Load graph data from a JSON file into Neo4j."""
    try:
        print(f"\nLoading graph data from {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict) or 'nodes' not in data or 'edges' not in data:
            print("Error: Invalid graph data format")
            return False
        
        # Generate a unique batch ID for this load
        batch_id = f"load_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Store all nodes and edges
        success = store_batch_in_neo4j(
            nodes=data['nodes'],
            edges=data['edges'],
            batch_id=batch_id
        )
        
        if success:
            print(f"\nSuccessfully loaded {len(data['nodes'])} nodes and {len(data['edges'])} edges")
            return True
        else:
            print("Error: Failed to store data in Neo4j")
            return False
            
    except Exception as e:
        print(f"Error loading graph data: {str(e)}")
        return False

if __name__ == "__main__":
        verify_neo4j_data()
        print_visualization_queries()