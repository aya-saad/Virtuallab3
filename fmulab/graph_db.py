"""
Neo4j database connection and query execution for FMU simulation
Fixed to handle empty results properly
"""
import logging
import json
from neo4j import GraphDatabase, time

class Neo4jConnection:
    def __init__(self, uri=None, username=None, password=None, database="neo4j"):
        """
        Initialize the Neo4j connection with credentials
        """
        from django.conf import settings
        self.uri = uri if uri is not None else settings.NEO4J_URI
        self.username = username if username is not None else settings.NEO4J_USERNAME
        self.password = password if password is not None else settings.NEO4J_PASSWORD
        self.database = database if database is not None else settings.NEO4J_DATABASE

        self.enable_user_agent = getattr(settings, "ENABLE_USER_AGENT", False)
        self.user_agent = getattr(settings, "NEO4J_USER_AGENT", None)

        # For debugging
        logging.info(f"Neo4jConnection initialized with URI: {self.uri}")
        logging.info(f"Using username: {self.username}")
        logging.info(f"Using database: {self.database}")
        self.driver = None

    def connect(self):
        """
        Creates and returns a Neo4j database driver instance
        """
        try:
            # Clean up URI - important for Aura connections
            uri = self.uri.strip()
            logging.info(f"Attempting to connect to the Neo4j database at {uri}")
            logging.info(f"Using username: {self.username}")
            logging.info(f"Using database: {self.database}")

            # Create the driver with appropriate settings
            if self.enable_user_agent and self.user_agent:
                self.driver = GraphDatabase.driver(
                    uri,
                    auth=(self.username, self.password),
                    database=self.database,
                    user_agent=self.user_agent
                )
            else:
                self.driver = GraphDatabase.driver(
                    uri,
                    auth=(self.username, self.password),
                    database=self.database
                )

            logging.info("Connection to Neo4j successful")
            return self.driver

        except Exception as e:
            error_message = f"Failed to connect to Neo4j at {uri}. Error: {str(e)}"
            logging.error(error_message, exc_info=True)
            return None

    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()

    def execute_query(self, query, params=None):
        """
        Executes a specified query using the Neo4j driver with proper error handling.

        Returns:
        tuple: Contains records, summary of the execution, and keys of the records.
        """
        if not self.driver:
            self.connect()

        if self.driver is None:
            raise Exception("Failed to establish Neo4j connection. Check credentials and connection.")

        try:
            # For newer Neo4j versions (4.0+) use execute_query method
            if hasattr(self.driver, 'execute_query'):
                # This returns a tuple of (records, summary, keys)
                return self.driver.execute_query(query, **(params or {}))
            else:
                # Fallback to session.run for older Neo4j versions
                with self.driver.session(database=self.database) as session:
                    result = session.run(query, params or {})
                    records = list(result)
                    summary = result.consume()
                    keys = result.keys() if records else []
                    return records, summary, keys

        except Exception as e:
            error_message = f"Failed to execute query: {str(e)}"
            logging.error(error_message, exc_info=True)
            raise Exception(error_message)

    def get_completed_documents(self):
        """
        Retrieves the names of all documents from the database.
        Fixed to handle empty results properly.
        """
        docs_query = "MATCH(node:Document) RETURN node"

        try:
            logging.info("Executing query to retrieve completed documents.")
            records, summary, keys = self.execute_query(docs_query)
            logging.info(f"Query executed successfully, retrieved {len(records)} records.")

            # Handle empty results
            if not records:
                return []

            # Extract document names based on Neo4j driver version and response format
            documents = []
            for record in records:
                if hasattr(record, "get") and "node" in record:
                    # New Neo4j driver format
                    if "fileName" in record["node"]:
                        documents.append(record["node"]["fileName"])
                elif record and len(record) > 0:
                    # Handle different format or older Neo4j driver
                    if "fileName" in record[0]:
                        documents.append(record[0]["fileName"])

            logging.info("Document names extracted successfully.")
            return documents

        except Exception as e:
            logging.error(f"An error occurred retrieving documents: {str(e)}")
            return []

    def get_graph_for_documents(self, document_names=None, query_text="", chunk_limit=50):
        """
        Get a graph representation for documents with relevance to a query.
        """
        try:
            base_query = """
            MATCH (c:Chunk)
            WHERE c.text CONTAINS $query_text
            WITH c, apoc.text.similarity(c.text, $query_text) AS relevance
            ORDER BY relevance DESC
            LIMIT $chunk_limit
            MATCH (c)-[:PART_OF]->(d:Document)
            OPTIONAL MATCH (c)-[:DISCUSSES]->(concept:Concept)
            RETURN c.text AS chunk_text, 
                   d.fileName AS source,
                   collect(DISTINCT concept.name) AS concepts
            """

            params = {
                "query_text": query_text,  # Use the actual query text
                "chunk_limit": chunk_limit
            }

            if document_names:
                # If specific documents are requested, add that filter
                base_query = base_query.replace(
                    "MATCH (c:Chunk)",
                    "MATCH (c:Chunk)-[:PART_OF]->(d:Document) WHERE d.fileName IN $document_names"
                )
                params["document_names"] = document_names

            # Execute query
            records, _, _ = self.execute_query(base_query, params)

            # Process results to include concepts
            result = {
                "chunks": [
                    {
                        "text": record.get("chunk_text", ""),
                        "source": record.get("source", "Unknown"),
                        "concepts": record.get("concepts", [])
                    }
                    for record in records if record.get("chunk_text")
                ]
            }

            return result
        except Exception as e:
            logging.error(f"Error retrieving graph: {str(e)}")
            return {"chunks": []}