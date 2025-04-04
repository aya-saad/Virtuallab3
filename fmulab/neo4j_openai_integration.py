"""
Enhanced QA Integration module for connecting Neo4j with OpenAI API
Updated to utilize Concept nodes in the knowledge graph
"""
import logging
import re
import json
import os
from datetime import datetime
import openai

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Constants for prompt engineering
SYSTEM_PROMPT = """
You are an FMU Simulation Assistant, specialized in aquaculture research and simulations. 
Your knowledge comes from a graph database that contains information about fish growth models, 
water treatment, hydrodynamic models, and more.

When answering questions, use the graph context provided below which includes:
- Content from relevant documents and chunks
- Content from relevant documents and chunks
- Concepts from the knowledge graph
- Relationships between chunks, documents, and concepts

### Guidelines:
1. Provide detailed, accurate answers based on the provided graph context.
2. Explain connections between concepts when relevant.
3. If the context contains different perspectives or approaches, summarize them.
4. If information on a specific topic is not found in the context, acknowledge this limitation.

### Graph Context:
{}

If you need to reference specific concepts from the knowledge graph, they are listed below:
{}

Remember to use the concepts and their relationships to provide a comprehensive answer.
"""

class OpenAIClient:
    """Wrapper for OpenAI API"""
    
    def __init__(self, api_key=None, model="gpt-3.5-turbo"):
        """Initialize the OpenAI client"""
        from django.conf import settings
        self.api_key = api_key or settings.OPENAI_API_KEY # os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            logging.warning("No OpenAI API key provided. Please set OPENAI_API_KEY environment variable.")
        
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)
        logging.info(f"OpenAI client initialized with model: {self.model}")
    
    def generate_response(self, messages, max_tokens=1000, temperature=0.0):
        """Generate a response using OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error generating response with OpenAI: {str(e)}")
            return f"I encountered an error connecting to my knowledge base: {str(e)}"


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
    
    def get_completed_documents(self):
        """
        Get list of completed documents from Neo4j
        Modified to handle empty results safely
        """
        try:
            # Simple query to get document names
            docs_query = "MATCH (node:Document) RETURN node.fileName AS fileName"
            
            logging.info("Executing query to retrieve documents.")
            records, _, _ = self.neo4j.execute_query(docs_query)
            logging.info(f"Query executed successfully, retrieved {len(records)} documents.")
            
            if not records:
                return []
                
            documents = [record.get("fileName", "Unknown") for record in records if record.get("fileName")]
            return documents
            
        except Exception as e:
            logging.error(f"An error occurred retrieving documents: {str(e)}")
            return []

    def retrieve_relevant_concepts(self, query, limit=10):
        """
        Retrieve relevant concepts from the knowledge graph based on the query
        """
        try:
            # Query to find relevant concepts
            concept_query = """
            // Find concepts that might be relevant to the query
            MATCH (c:Concept)
            WHERE toLower(c.name) CONTAINS toLower($query_text)
               OR toLower(c.description) CONTAINS toLower($query_text)
            
            // Calculate relevance score
            WITH c, 
                 CASE 
                     WHEN toLower(c.name) CONTAINS toLower($query_text) THEN 3
                     WHEN toLower(c.description) CONTAINS toLower($query_text) THEN 2
                     ELSE 1
                 END AS relevance
            ORDER BY relevance DESC
            LIMIT $limit
            
            // Return concept information
            RETURN c.name AS name, 
                   c.description AS description,
                   c.category AS category
            """
            
            params = {
                "query_text": query,
                "limit": limit
            }
            
            records, _, _ = self.neo4j.execute_query(concept_query, params)
            
            if not records:
                logging.info(f"No relevant concepts found for query: {query}")
                return []
            
            # Format the concepts into a structured list
            concepts = []
            for record in records:
                concept = {
                    "name": record.get("name", ""),
                    "description": record.get("description", ""),
                    "category": record.get("category", "")
                }
                if concept["name"]:
                    concepts.append(concept)
            
            return concepts
            
        except Exception as e:
            logging.error(f"Error retrieving concepts: {str(e)}")
            return []

    def retrieve_graph_context(self, query, limit=10):
        """
        Retrieve relevant information from Neo4j based on the query
        Updated to leverage the KAG structure with Concept nodes
        """
        try:
            # Query to find chunks and related concepts
            context_query = """
            // Find chunks that match the query text
            MATCH (c:Chunk)
            WHERE toLower(c.text) CONTAINS toLower($query_text)
            
            // Calculate relevance score
            WITH c, apoc.text.similarity(toLower(c.text), toLower($query_text)) AS relevance
            ORDER BY relevance DESC
            LIMIT $limit
            
            // Get document information
            OPTIONAL MATCH (c)-[:PART_OF]->(d:Document)
            
            // Get concepts related to chunks via DISCUSSES
            OPTIONAL MATCH (c)-[:DISCUSSES]->(concept:Concept)
            
            // Get related chunks via NEXT_CHUNK
            OPTIONAL MATCH (c)-[:NEXT_CHUNK]-(related_chunk:Chunk)
            
            // Return context information
            RETURN c.text AS chunk_text,
                   d.fileName AS source,
                   collect(DISTINCT {
                       name: concept.name,
                       description: concept.description,
                       category: concept.category
                   }) AS concepts,
                   collect(DISTINCT related_chunk.text) AS related_chunks
            """
            
            params = {
                "query_text": query,
                "limit": limit
            }
            
            records, _, _ = self.neo4j.execute_query(context_query, params)
            
            if not records:
                logging.info(f"No relevant chunks found for query: {query}")
                return "No relevant information found in the knowledge graph.", []
            
            # Format the results into context
            context_parts = []
            all_concepts = []
            
            for record in records:
                chunk_text = record.get("chunk_text", "")
                source = record.get("source", "Unknown")
                concepts = record.get("concepts", [])
                related_chunks = record.get("related_chunks", [])
                
                # Extract valid concepts
                valid_concepts = [c for c in concepts if c.get("name")]
                all_concepts.extend(valid_concepts)
                
                if chunk_text:
                    part = f"Source: {source}\n\nContent: {chunk_text}"
                    
                    # Add related concepts if any
                    if valid_concepts:
                        concept_names = [c.get("name", "") for c in valid_concepts]
                        part += f"\n\nRelated concepts: {', '.join(concept_names)}"
                    
                    # Add related chunks if any (truncated if too long)
                    if related_chunks:
                        related_text = "\n".join([chunk[:200] + "..." if len(chunk) > 200 else chunk 
                                                 for chunk in related_chunks[:3]])
                        part += f"\n\nRelated content: {related_text}"
                    
                    context_parts.append(part)
            
            # Deduplicate concepts by name
            unique_concepts = []
            seen_names = set()
            for concept in all_concepts:
                name = concept.get("name", "")
                if name and name not in seen_names:
                    seen_names.add(name)
                    unique_concepts.append(concept)
            
            return "\n\n---\n\n".join(context_parts), unique_concepts
            
        except Exception as e:
            logging.error(f"Error retrieving graph context: {str(e)}")
            return f"Error accessing knowledge graph: {str(e)}", []

    def format_concepts_for_prompt(self, concepts):
        """Format concepts into a readable format for the prompt"""
        if not concepts:
            return "No specific concepts were found relevant to your query."
            
        formatted_concepts = []
        for concept in concepts:
            name = concept.get("name", "")
            description = concept.get("description", "")
            category = concept.get("category", "")
            
            if name:
                concept_str = f"- {name}"
                if description:
                    concept_str += f": {description}"
                if category:
                    concept_str += f" (Category: {category})"
                formatted_concepts.append(concept_str)
        
        return "\n".join(formatted_concepts)

    def get_chat_response(self, query, session_id=None):
        """
        Get a response to the user query using the Neo4j knowledge graph and OpenAI
        Updated to utilize concepts from the KAG
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
            
            # Retrieve context and concepts from Neo4j
            context, concepts = self.retrieve_graph_context(query)
            
            # If no concepts were found via chunks, try direct concept retrieval
            if not concepts:
                concepts = self.retrieve_relevant_concepts(query)
            
            # Format concepts for the prompt
            formatted_concepts = self.format_concepts_for_prompt(concepts)
            
            # Extract sources for attribution
            sources = []
            source_matches = re.findall(r"Source: (.+?)\n", context)
            if source_matches:
                sources = list(set(source_matches))
            
            # Prepare messages for OpenAI, including system prompt with context and concepts
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT.format(context, formatted_concepts)},
                {"role": "user", "content": query}
            ]
            
            # Generate response with OpenAI
            response = self.openai.generate_response(messages)
            
            # Add the assistant's response to chat history
            self.chat_history[session_id].append({"role": "assistant", "content": response})
            
            return {
                "message": response,
                "sources": sources,
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
    
    def close(self):
        """Close the Neo4j connection"""
        if hasattr(self, 'neo4j') and self.neo4j:
            self.neo4j.close()
