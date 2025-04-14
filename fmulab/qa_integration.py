"""
QA Integration module for connecting Neo4j with LLMs
"""
import logging
import re
import json
import os
from datetime import datetime
from .graph_db import Neo4jConnection
from .llm_integration import get_llm_provider
from .neo4j_openai_integration import *

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class QAIntegration:
    """Enhanced QA Pipeline integrating Neo4j and OpenAI"""

    def __init__(self, neo4j_connection=None, openai_api_key=None, openai_model="gpt-3.5-turbo"):
        """Initialize the QA pipeline"""
        from django.conf import settings

        # Initialize Neo4j connection
        if neo4j_connection is None:
            from .graph_db import Neo4jConnection
            self.neo4j = Neo4jConnection(
                uri=settings.NEO4J_URI,
                username=settings.NEO4J_USERNAME,
                password=settings.NEO4J_PASSWORD,
                database=settings.NEO4J_DATABASE
            )
        else:
            self.neo4j = neo4j_connection

        # Connect to Neo4j if not already connected
        if not self.neo4j.driver:
            self.neo4j.connect()

        # Initialize OpenAI client
        self.openai = OpenAIClient(api_key=openai_api_key, model=openai_model)

        # Initialize chat history storage
        self.chat_history = {}

    def get_documents(self):
        """Get list of available documents"""
        return self.neo4j.get_completed_documents()

    def process_map_reduce_chain(self, query):
        """
        Process a query through the map-reduce chain
        Works with Neo4j Aura Free (no APOC needed)
        """
        try:
            # Step 1: Get relevant chunks from Neo4j using basic text matching
            # Breaking the query into terms for better matching
            query_terms = [term.lower() for term in query.split() if len(term) > 3]

            # Build a more flexible query for Neo4j Aura
            where_clauses = []
            for term in query_terms:
                # Sanitize the term for Cypher query
                sanitized_term = term.replace("'", "''")
                where_clauses.append(f"toLower(c.text) CONTAINS '{sanitized_term}'")

            # If no terms to search for, use the whole query
            if not where_clauses:
                sanitized_query = query.replace("'", "''")
                where_clause = f"toLower(c.text) CONTAINS toLower('{sanitized_query}')"
            else:
                where_clause = " OR ".join(where_clauses)

            chunk_query = f"""
            MATCH (c:Chunk)
            WHERE {where_clause}
            RETURN c.id AS id, c.text AS text, 
                   c.position AS position
            ORDER BY position
            LIMIT 10
            """

            records, _, _ = self.neo4j.driver.execute_query(chunk_query)

            if not records:
                return "I couldn't find relevant information to answer your question."

            # Step 2: Get document info and concepts for chunks
            chunks_with_metadata = []
            for record in records:
                chunk_id = record.get("id", "")

                # Get document info in one query
                doc_query = """
                MATCH (c:Chunk {id: $chunk_id})-[:PART_OF]->(d:Document)
                RETURN d.fileName AS source
                """

                doc_records, _, _ = self.neo4j.driver.execute_query(doc_query, {"chunk_id": chunk_id})

                # Get concepts in another query
                concept_query = """
                MATCH (c:Chunk {id: $chunk_id})-[:DISCUSSES]->(concept:Concept)
                RETURN collect(concept.name) AS concepts
                """

                concept_records, _, _ = self.neo4j.driver.execute_query(concept_query, {"chunk_id": chunk_id})

                source = "Unknown"
                concepts = []

                if doc_records:
                    source = doc_records[0].get("source", "Unknown")

                if concept_records:
                    concepts = concept_records[0].get("concepts", [])

                # Add all data to chunk
                chunks_with_metadata.append({
                    "id": chunk_id,
                    "text": record.get("text", ""),
                    "position": record.get("position", 0),
                    "source": source,
                    "concepts": concepts
                })

            # Step 3: Process each chunk through map step
            mapped_results = []
            for chunk in chunks_with_metadata:
                # Format the chunk data for the map step
                chunk_data = f"""
                Chunk ID: {chunk['id']}
                Source: {chunk['source']}
                Position: {chunk['position']}
                Concepts: {', '.join(chunk['concepts']) if chunk['concepts'] else 'None'}
                
                Content:
                {chunk['text']}
                """

                # Process this chunk using the OpenAI client
                map_result = self.openai.map_chunk(chunk_data, query)
                mapped_results.append(map_result)

            # Step 4: Combine all mapped results in reduce step
            final_response = self.openai.reduce_results(mapped_results, query)

            return final_response

        except Exception as e:
            logging.error(f"Error in map-reduce chain: {e}")
            return f"An error occurred while processing your question: {str(e)}"

    def get_chat_response(self, query, session_id=None, document_names=None, chat_mode=None):
        """
        Get a response to the user query using the Neo4j knowledge graph and OpenAI
        Using Map-Reduce approach for better results
        """
        try:
            # Create or get session ID
            if not session_id:
                session_id = f"session_{datetime.now().timestamp()}"

            # Initialize chat history for new sessions
            if session_id not in self.chat_history:
                self.chat_history[session_id] = []

            # Add the user's query to chat history
            self.chat_history[session_id].append({"role": "user", "content": query})

            # Process the query through map-reduce chain
            response_content = self.process_map_reduce_chain(query)

            # Extract sources for attribution
            sources = []
            # Look for sources mentioned in the format "Source: filename"
            source_matches = re.findall(r"Source: ([^\n]+)", response_content)
            if source_matches:
                sources = list(set(source_matches))

            # Also try to find Data: Chunk references
            chunk_matches = re.findall(r"\[Data: Chunk \(([^\)]+)\)\]", response_content)
            if chunk_matches:
                for chunk_id in chunk_matches:
                    # Try to find the source for this chunk
                    try:
                        doc_query = """
                        MATCH (c:Chunk {id: $chunk_id})-[:PART_OF]->(d:Document)
                        RETURN d.fileName AS source
                        """

                        doc_records, _, _ = self.neo4j.driver.execute_query(doc_query, {"chunk_id": chunk_id})

                        if doc_records and doc_records[0].get("source"):
                            sources.append(doc_records[0].get("source"))
                    except Exception as source_error:
                        logging.error(f"Error finding source for chunk {chunk_id}: {source_error}")

            # Add the assistant's response to chat history
            self.chat_history[session_id].append({"role": "assistant", "content": response_content})

            return {
                "message": response_content,
                "sources": list(set(sources)),  # Deduplicate sources
                "session_id": session_id
            }

        except Exception as e:
            logging.error(f"Error in QA integration: {str(e)}", exc_info=True)
            error_message = f"I'm sorry, but I encountered an error processing your question: {str(e)}"

            # Add error response to chat history
            if session_id in self.chat_history:
                self.chat_history[session_id].append({"role": "assistant", "content": error_message})

            return {
                "message": error_message,
                "sources": [],
                "session_id": session_id
            }