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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Constants for prompt engineering
SYSTEM_PROMPT = """
You are an FMU Simulation Assistant, specialized in aquaculture research and simulations. Your knowledge comes from a graph database that contains information about fish growth models, water treatment, hydrodynamic models, and more.

When answering questions, use the graph context provided below which includes:
- Content from relevant documents
- Entities (concepts, terms, components) from the knowledge graph
- Relationships between these entities

### Guidelines:
1. Provide detailed, accurate answers based solely on the provided graph context.
2. Explain connections between concepts when relevant.
3. If the context contains different perspectives or approaches, summarize them.
4. Be specific about what files or components you're referring to.
5. If the graph context doesn't contain enough information, acknowledge this limitation.

### Graph Context:
{}

Remember: You have access to a rich knowledge graph about FMU simulations and aquaculture research. Use this specialized knowledge to provide helpful, accurate responses.
"""



class QAIntegration:
    """QA Pipeline integrating Neo4j and LLMs"""

    def __init__(self, neo4j_connection=None, llm_provider=None):
        """Initialize the QA pipeline"""
        from django.conf import settings

        if neo4j_connection is None:
            # Explicitly pass the settings from Django
            self.neo4j = Neo4jConnection(
                uri=settings.NEO4J_URI,
                username=settings.NEO4J_USERNAME,
                password=settings.NEO4J_PASSWORD,
                database=settings.NEO4J_DATABASE
            )
        else:
            self.neo4j = neo4j_connection

        if not self.neo4j.driver:
            self.neo4j.connect()

        self.llm = llm_provider or get_llm_provider()
        self.chat_history = {}  # Session ID -> list of messages

    def get_documents(self):
        """Get list of available documents"""
        return self.neo4j.get_completed_documents()

    def _format_context_from_chunks(self, chunks):
        """Format chunks into context for LLM prompt"""
        formatted_chunks = []

        for i, chunk in enumerate(chunks):
            formatted_chunk = f"Document {i + 1}:\n{chunk.get('text', '')}\n"
            formatted_chunks.append(formatted_chunk)

        return "\n\n".join(formatted_chunks)

    def retrieve_chunks(self, query, document_names=None, limit=5):
        """Retrieve relevant chunks from Neo4j based on query"""
        try:
            # Basic semantic search query - in production, use vector similarity
            search_query = """
            MATCH (d:Document)
            WHERE d.fileName IN $document_names
            MATCH (d)<-[:PART_OF]-(c:Chunk)
            WHERE c.text CONTAINS $query_text
            RETURN c
            LIMIT $limit
            """

            if not document_names:
                # If no documents specified, search all documents
                search_query = """
                MATCH (d:Document)<-[:PART_OF]-(c:Chunk)
                WHERE toLower(c.text) CONTAINS toLower($query_text)
                RETURN c
                LIMIT $limit
                """

            params = {
                "query_text": query,
                "limit": limit
            }

            if document_names:
                params["document_names"] = document_names

            records, _ = self.neo4j.execute_query(search_query, params)

            chunks = []
            for record in records:
                chunk = record["c"]
                chunks.append({
                    "id": chunk.get("id", ""),
                    "text": chunk.get("text", ""),
                    "source": record.get("fileName", "Unknown")
                })

            return chunks

        except Exception as e:
            logging.error(f"Error retrieving chunks: {str(e)}")
            return []

    def retrieve_graph_context(self, query, document_names=None, limit=10):
        """Retrieve relevant graph context from Neo4j based on query"""
        try:
            # Use a more sophisticated query that leverages the graph structure
            # This query finds relevant chunks, then expands to include connected entities and their relationships
            graph_query = """
            // First find relevant chunks using vector similarity or text matching
            MATCH (c:Chunk)
            WHERE c.text CONTAINS $query_text OR EXISTS(c.embedding)
            WITH c, CASE 
                WHEN EXISTS(c.embedding) THEN 1.0 
                ELSE apoc.text.similarity(c.text, $query_text) 
            END AS relevance
            ORDER BY relevance DESC
            LIMIT $limit

            // Get the documents these chunks belong to
            MATCH (c)-[:PART_OF]->(d:Document)

            // Expand to related entities
            OPTIONAL MATCH path = (c)-[:HAS_ENTITY]->(e)

            // Also get relationships between entities
            OPTIONAL MATCH entity_rels = (e)-[r]-(other:__Entity__)

            // Collect all the results
            RETURN c.text AS chunk_text, 
                   d.fileName AS source,
                   collect(DISTINCT e.id) AS entities,
                   collect(DISTINCT type(r) + ': ' + other.id) AS relationships
            """

            params = {
                "query_text": query,
                "limit": limit
            }

            if document_names:
                graph_query = graph_query.replace(
                    "MATCH (c:Chunk)",
                    "MATCH (c:Chunk)-[:PART_OF]->(d:Document) WHERE d.fileName IN $document_names"
                )
                params["document_names"] = document_names

            records, _, _ = self.neo4j.execute_query(graph_query, params)

            # Process the results into a context format
            context_parts = []

            for record in records:
                chunk_text = record["chunk_text"]
                source = record["source"]
                entities = record["entities"]
                relationships = record["relationships"]

                context_part = f"Source: {source}\n\nContent: {chunk_text}\n"

                if entities:
                    context_part += "\nEntities: " + ", ".join(entities)

                if relationships:
                    filtered_rels = [r for r in relationships if r is not None]
                    if filtered_rels:
                        context_part += "\nRelationships: " + ", ".join(filtered_rels)

                context_parts.append(context_part)

            return context_parts

        except Exception as e:
            logging.error(f"Error retrieving graph context: {str(e)}")
            return []

    def get_chat_response(self, query, session_id=None, document_names=None):
        """Get a response to the user query"""
        try:
            # Initialize or get chat history
            if not session_id:
                session_id = f"session_{datetime.now().timestamp()}"

            if session_id not in self.chat_history:
                self.chat_history[session_id] = []

            # Add the query to chat history
            self.chat_history[session_id].append({"role": "user", "content": query})

            # Retrieve relevant chunks
            # chunks = self.retrieve_chunks(query, document_names)
            # Retrieve relevant graph context
            context_parts = self.retrieve_graph_context(query, document_names)

            #if not chunks:
            if not context_parts:
                no_info_response = "I couldn't find any relevant information to answer your question."
                self.chat_history[session_id].append({"role": "assistant", "content": no_info_response})
                return {
                    "message": no_info_response,
                    "sources": [],
                    "session_id": session_id
                }

            # Format chunks into context
            #context = self._format_context_from_chunks(chunks)
            # Format context for the prompt
            context = "\n\n---\n\n".join(context_parts)

            # Generate the prompt with context
            prompt = SYSTEM_PROMPT.format(context) + f"\nQuestion: {query}"

            # Get response from LLM
            logging.info(f"Sending prompt to LLM: {prompt[:200]}...")  # Log first 200 chars
            response = self.llm.generate(prompt)
            logging.info(f"Received response from LLM: {response[:200]}...")  # Log first 200 chars

            # Add response to chat history
            self.chat_history[session_id].append({"role": "assistant", "content": response})

            # Extract sources for attribution
            # sources = list(set([chunk.get("source", "Unknown") for chunk in chunks]))
            sources = []
            for part in context_parts:
                source_match = re.search(r"Source: (.+?)\n", part)
                if source_match and source_match.group(1) not in sources:
                    sources.append(source_match.group(1))


            return {
                "message": response,
                "sources": sources,
                "session_id": session_id
            }

        except Exception as e:
            logging.error(f"Error in QA pipeline: {str(e)}", exc_info=True)
            error_response = f"I'm sorry, but I encountered an error while processing your question: {str(e)}"

            if session_id in self.chat_history:
                self.chat_history[session_id].append({"role": "assistant", "content": error_response})

            return {
                "message": error_response,
                "sources": [],
                "session_id": session_id
            }

    def close(self):
        """Close connections"""
        if self.neo4j:
            self.neo4j.close()