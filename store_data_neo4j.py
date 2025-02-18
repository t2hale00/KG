import os
import re
import json
from typing import List, Dict, Tuple
from pypdf import PdfReader
from neo4j import GraphDatabase
import hashlib

# Neo4j Configuration (from environment variables)
URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

# Cache file for Neo4j state
NEO4J_CACHE_FILE = "neo4j_cache.json"

from neo4j import GraphDatabase

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