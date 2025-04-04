"""
Configuration settings for the Graph LLM integration
"""
import os

# Neo4j connection settings
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")
ENABLE_USER_AGENT = os.environ.get("ENABLE_USER_AGENT", "False").lower() in ("true", "1", "yes")
NEO4J_USER_AGENT = os.environ.get("NEO4J_USER_AGENT", "neo4j-python/4.4.0")

# LLM settings
DEFAULT_LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")  # openai or azure
DEFAULT_LLM_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
DEFAULT_MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1000"))
DEFAULT_TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))

# Graph chunk limit for queries
GRAPH_CHUNK_LIMIT = int(os.environ.get("GRAPH_CHUNK_LIMIT", "50"))

# Chat modes - based on the original constants.py
CHAT_DEFAULT_MODE = "graph_vector_fulltext"  # Default from original code
CHAT_VECTOR_MODE = "vector"
CHAT_FULLTEXT_MODE = "fulltext"
CHAT_ENTITY_VECTOR_MODE = "entity_vector"
CHAT_VECTOR_GRAPH_MODE = "graph_vector"
CHAT_VECTOR_GRAPH_FULLTEXT_MODE = "graph_vector_fulltext"
CHAT_GLOBAL_VECTOR_FULLTEXT_MODE = "global_vector"
CHAT_GRAPH_MODE = "graph"
CHAT_SIMULATION_MODE = "simulation_science"

# This is the main GRAPH_QUERY used by get_graph_for_documents
# Directly from original constants.py
GRAPH_QUERY = """
MATCH docs = (d:Document) 
WHERE d.fileName IN $document_names
WITH docs, d 
ORDER BY d.createdAt DESC

// Fetch chunks for documents, currently with limit
CALL {{
  WITH d
  OPTIONAL MATCH chunks = (d)<-[:PART_OF|FIRST_CHUNK]-(c:Chunk)
  RETURN c, chunks LIMIT {graph_chunk_limit}
}}

WITH collect(distinct docs) AS docs, 
     collect(distinct chunks) AS chunks, 
     collect(distinct c) AS selectedChunks

// Select relationships between selected chunks
WITH *, 
     [c IN selectedChunks | 
       [p = (c)-[:NEXT_CHUNK|SIMILAR]-(other) 
       WHERE other IN selectedChunks | p]] AS chunkRels

// Fetch entities and relationships between entities
CALL {{
  WITH selectedChunks
  UNWIND selectedChunks AS c
  OPTIONAL MATCH entities = (c:Chunk)-[:HAS_ENTITY]->(e)
  OPTIONAL MATCH entityRels = (e)--(e2:!Chunk) 
  WHERE exists {{
    (e2)<-[:HAS_ENTITY]-(other) WHERE other IN selectedChunks
  }}
  RETURN entities, entityRels, collect(DISTINCT e) AS entity
}}

WITH docs, chunks, chunkRels, 
     collect(entities) AS entities, 
     collect(entityRels) AS entityRels, 
     entity

WITH *

CALL {{
  WITH entity
  UNWIND entity AS n
  OPTIONAL MATCH community = (n:__Entity__)-[:IN_COMMUNITY]->(p:__Community__)
  OPTIONAL MATCH parentcommunity = (p)-[:PARENT_COMMUNITY*]->(p2:__Community__) 
  RETURN collect(community) AS communities, 
         collect(parentcommunity) AS parentCommunities
}}

WITH apoc.coll.flatten(docs + chunks + chunkRels + entities + entityRels + communities + parentCommunities, true) AS paths

// Distinct nodes and relationships
CALL {{
  WITH paths 
  UNWIND paths AS path 
  UNWIND nodes(path) AS node 
  WITH distinct node 
  RETURN collect(node /* {{.*, labels:labels(node), elementId:elementId(node), embedding:null, text:null}} */) AS nodes 
}}

CALL {{
  WITH paths 
  UNWIND paths AS path 
  UNWIND relationships(path) AS rel 
  RETURN collect(distinct rel) AS rels 
}}  

RETURN nodes, rels
"""

# Search query templates
VECTOR_SEARCH_QUERY = """
WITH node AS chunk, score
MATCH (chunk)-[:PART_OF]->(d:Document)
WITH d, 
     collect(distinct {chunk: chunk, score: score}) AS chunks, 
     avg(score) AS avg_score

WITH d, avg_score, 
     [c IN chunks | c.chunk.text] AS texts, 
     [c IN chunks | {id: c.chunk.id, score: c.score}] AS chunkdetails

WITH d, avg_score, chunkdetails, 
     apoc.text.join(texts, "\n----\n") AS text

RETURN text, 
       avg_score AS score, 
       {source: COALESCE(CASE WHEN d.url CONTAINS "None" 
                             THEN d.fileName 
                             ELSE d.url 
                       END, 
                       d.fileName), 
        chunkdetails: chunkdetails} AS metadata
"""

VECTOR_GRAPH_SEARCH_QUERY_PREFIX = """
WITH node as chunk, score
// find the document of the chunk
MATCH (chunk)-[:PART_OF]->(d:Document)
// aggregate chunk-details
WITH d, collect(DISTINCT {chunk: chunk, score: score}) AS chunks, avg(score) as avg_score
// fetch entities
CALL { WITH chunks
UNWIND chunks as chunkScore
WITH chunkScore.chunk as chunk
"""

VECTOR_GRAPH_SEARCH_ENTITY_QUERY = """
    OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e)
    WITH e, count(*) AS numChunks 
    ORDER BY numChunks DESC 
    LIMIT {no_of_entites}

    WITH 
    CASE 
        WHEN e.embedding IS NULL OR ({embedding_match_min} <= vector.similarity.cosine($embedding, e.embedding) AND vector.similarity.cosine($embedding, e.embedding) <= {embedding_match_max}) THEN 
            collect {{
                OPTIONAL MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){{0,1}}(:!Chunk&!Document&!__Community__) 
                RETURN path LIMIT {entity_limit_minmax_case}
            }}
        WHEN e.embedding IS NOT NULL AND vector.similarity.cosine($embedding, e.embedding) >  {embedding_match_max} THEN
            collect {{
                OPTIONAL MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){{0,2}}(:!Chunk&!Document&!__Community__) 
                RETURN path LIMIT {entity_limit_max_case} 
            }} 
        ELSE 
            collect {{ 
                MATCH path=(e) 
                RETURN path 
            }}
    END AS paths, e
"""

# Constants for vector graph search
VECTOR_GRAPH_SEARCH_ENTITY_LIMIT = 40
VECTOR_GRAPH_SEARCH_EMBEDDING_MIN_MATCH = 0.3
VECTOR_GRAPH_SEARCH_EMBEDDING_MAX_MATCH = 0.9
VECTOR_GRAPH_SEARCH_ENTITY_LIMIT_MINMAX_CASE = 20
VECTOR_GRAPH_SEARCH_ENTITY_LIMIT_MAX_CASE = 40

# Vector search parameters
VECTOR_SEARCH_TOP_K = 5

# Mode configuration mapping - based on original CHAT_MODE_CONFIG_MAP
CHAT_MODE_CONFIG_MAP= {
    CHAT_VECTOR_MODE: {
        "retrieval_query": VECTOR_SEARCH_QUERY,
        "top_k": VECTOR_SEARCH_TOP_K,
        "index_name": "vector",
        "keyword_index": None,
        "document_filter": True,
        "node_label": "Chunk",
        "embedding_node_property":"embedding",
        "text_node_properties":["text"],
        "description": "Simple text search on chunks"
    },
    CHAT_FULLTEXT_MODE: {
        "retrieval_query": VECTOR_SEARCH_QUERY,
        "top_k": VECTOR_SEARCH_TOP_K,
        "index_name": "vector",
        "keyword_index": "keyword",
        "document_filter": False,
        "node_label": "Chunk",
        "embedding_node_property":"embedding",
        "text_node_properties":["text"],
        "description": "Full-text search with keywords"
    },
    CHAT_ENTITY_VECTOR_MODE: {
        "retrieval_query": VECTOR_SEARCH_QUERY,
        "top_k": VECTOR_SEARCH_TOP_K,
        "index_name": "entity_vector",
        "keyword_index": None,
        "document_filter": False,
        "node_label": "__Entity__",
        "embedding_node_property":"embedding",
        "text_node_properties":["id"],
        "description": "Entity-focused vector search"
    },
    CHAT_VECTOR_GRAPH_MODE: {
        "retrieval_query": VECTOR_SEARCH_QUERY,
        "top_k": VECTOR_SEARCH_TOP_K,
        "index_name": "vector",
        "keyword_index": None,
        "document_filter": True,
        "node_label": "Chunk",
        "embedding_node_property":"embedding",
        "text_node_properties":["text"],
        "description": "Graph-based search with vector similarity"
    },
    CHAT_VECTOR_GRAPH_FULLTEXT_MODE: {
        "retrieval_query": VECTOR_SEARCH_QUERY,
        "top_k": VECTOR_SEARCH_TOP_K,
        "index_name": "vector",
        "keyword_index": "keyword",
        "document_filter": False,
        "node_label": "Chunk",
        "embedding_node_property":"embedding",
        "text_node_properties":["text"],
        "description": "Combined graph and full-text search"
    },
    CHAT_GLOBAL_VECTOR_FULLTEXT_MODE: {
        "retrieval_query": VECTOR_SEARCH_QUERY,
        "top_k": VECTOR_SEARCH_TOP_K,
        "index_name": "community_vector",
        "keyword_index": "community_keyword",
        "document_filter": False,
        "node_label": "__Community__",
        "embedding_node_property":"embedding",
        "text_node_properties":["summary"],
        "description": "Global vector search with communities"
    },
    CHAT_GRAPH_MODE: {
        "description": "Pure graph-based search",
        "top_k": 5,
    },
}