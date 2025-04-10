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
# In fmulab/neo4j_openai_integration.py, update the SYSTEM_PROMPT
SYSTEM_PROMPT = """
You are an FMU Simulation Assistant, specialized in aquaculture research and simulations. 
Your knowledge comes from a graph database that contains information about fish growth models, 
water treatment, hydrodynamic models, and more.

When answering questions:
1. ONLY use the information provided in the graph context below.
2. DO NOT make up or invent information not present in the context.
3. If specific information is not available in the context, acknowledge this limitation.
4. Be specific and detailed when the context provides detailed information.
5. Provide step-by-step instructions when available in the context.

### Graph Context:
{}

### Key Concepts from the Knowledge Graph:
{}

Remember to base your answer EXCLUSIVELY on the above context.
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

    # Then update the retrieve_graph_context method to better leverage the query
    def retrieve_graph_context(self, query, limit=10):
        """
        Retrieve relevant information from Neo4j based on the query
        Updated to leverage the KAG structure with Concept nodes and Community nodes
        """
        try:
            # Try to get community summaries first - these provide high-level context
            community_query = """
            MATCH (c:__Community__)<-[:IN_COMMUNITY]-(:__Entity__)<-[:MENTIONS]-(chunk:Chunk)
            WHERE chunk.text CONTAINS $query_text
            WITH c, count(distinct chunk) AS relevance, c.summary AS summary
            WHERE summary IS NOT NULL
            RETURN summary
            ORDER BY relevance DESC
            LIMIT 3
            """

            community_params = {
                "query_text": query
            }

            community_records, _, _ = self.neo4j.execute_query(community_query, community_params)
            community_context = [record.get("summary", "") for record in community_records]

            # Then get specific chunk matches
            chunk_query = """
            MATCH (c:Chunk)
            WHERE toLower(c.text) CONTAINS toLower($query_text)
            WITH c, apoc.text.similarity(toLower(c.text), toLower($query_text)) AS relevance
            ORDER BY relevance DESC
            LIMIT $limit

            MATCH (c)-[:PART_OF]->(d:Document)
            OPTIONAL MATCH (c)-[:DISCUSSES]->(concept:Concept)

            RETURN c.text AS chunk_text,
                   d.fileName AS source,
                   collect(DISTINCT concept.name) AS concepts
            """

            chunk_params = {
                "query_text": query,
                "limit": limit
            }

            chunk_records, _, _ = self.neo4j.execute_query(chunk_query, chunk_params)

            # Format results into context
            context_parts = []
            all_concepts = []

            # Add community summaries first
            if community_context:
                context_parts.append("# Community Knowledge\n\n" + "\n\n".join(community_context))

            # Add chunk context
            if chunk_records:
                context_parts.append("# Specific Document Chunks\n")

                for record in chunk_records:
                    chunk_text = record.get("chunk_text", "")
                    source = record.get("source", "Unknown")
                    concepts = record.get("concepts", [])

                    if chunk_text:
                        part = f"Source: {source}\n\nContent: {chunk_text}"

                        # Add related concepts if any
                        if concepts:
                            concept_names = [c for c in concepts if c]
                            if concept_names:
                                part += f"\n\nRelated concepts: {', '.join(concept_names)}"
                                all_concepts.extend(concept_names)

                        context_parts.append(part)

            # If we have no context at all, return a message
            if not context_parts:
                return "No relevant information found in the knowledge graph.", []

            # Deduplicate concepts
            unique_concepts = list(set(all_concepts))

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

            # Retrieve context and concepts from Neo4j - this now needs to include the query
            self.neo4j.query_text = query  # Modify your Neo4j class to store this
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
