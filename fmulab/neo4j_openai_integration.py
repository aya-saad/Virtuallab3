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

# Map-Reduce prompts for better analysis
MAP_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant specialized in fish growth modeling and aquaculture simulations.

---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data.

You should use the data provided below as the primary context for generating the response.
If you don't know the answer or if the input data does not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point, focusing on practical steps for simulation.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question.

The response should be JSON formatted as follows:
{{
    "points": [
        {{"description": "Description of point 1 [Data: Chunk (chunk_id)]", "score": score_value}},
        {{"description": "Description of point 2 [Data: Chunk (chunk_id)]", "score": score_value}}
    ]
}}

When explaining simulations:
- Provide specific parameter values when available
- Mention file names and paths when referenced
- Include step-by-step instructions for Kopl tool usage
- Explain configuration procedures clearly

---Data---

{context_data}
"""

REDUCE_SYSTEM_PROMPT = """
---Role---

You are an FMU Simulation Assistant, specialized in aquaculture research and simulations with the Kopl tool.

---Goal---

Generate a response that answers the user's question by synthesizing information from multiple sources. The response should be comprehensive, detailed, and focus on practical implementation steps.

Note that the information provided below is ranked in descending order of importance.

If you don't know the answer or if the provided information is insufficient, just say so. Do not make anything up.

The final response should be well-structured with:
- Clear steps for simulation setup
- Specific parameter values when available
- References to configuration files and tools
- Explanations of key concepts where needed

When explaining how to use the Kopl tool specifically:
1. Include details about creating co-simulation tasks
2. Explain how to load and configure XML files
3. Describe parameter settings with specific values
4. Outline running procedures and accessing results

Format your response in markdown with appropriate headers and lists for clarity.

---Information Sources---

{report_data}

---Response Format---

{response_type}
"""

class OpenAIClient:
    """Wrapper for OpenAI API"""

    def __init__(self, api_key=None, model="gpt-3.5-turbo"):
        """Initialize the OpenAI client"""
        from django.conf import settings
        self.api_key = api_key or settings.OPENAI_API_KEY
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

    def map_chunk(self, chunk_data, query):
        """Process a single chunk through the map step"""
        try:
            messages = [
                {"role": "system", "content": MAP_SYSTEM_PROMPT.format(context_data=chunk_data)},
                {"role": "user", "content": query}
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.0
            )

            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in map step: {e}")
            return json.dumps({"points": []})

    def reduce_results(self, mapped_results, query, response_type="multiple paragraphs"):
        """Combine mapped results through the reduce step"""
        try:
            messages = [
                {"role": "system", "content": REDUCE_SYSTEM_PROMPT.format(
                    report_data="\n\n".join(mapped_results),
                    response_type=response_type
                )},
                {"role": "user", "content": query}
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1500,
                temperature=0.0
            )

            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in reduce step: {e}")
            return "I couldn't process the information to answer your question."