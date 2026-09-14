"""
Amazon Bedrock client for embeddings and reasoning.
Handles Titan embeddings and Claude Sonnet 4.5 interactions.
"""

import boto3
import json
import os
from typing import List, Dict, Any, Optional


class BedrockClient:
    """Client for Amazon Bedrock services."""

    # Model IDs
    TITAN_EMBED_MODEL = "amazon.titan-embed-text-v2:0"
    CLAUDE_SONNET_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    EXPECTED_EMBEDDING_DIM = 1024

    def __init__(self, profile_name: Optional[str] = None):
        """
        Initialize Bedrock client.

        Args:
            profile_name: AWS profile name for local development.
                         If None, uses default credentials (e.g., Lambda IAM role).
        """
        session_kwargs = {}
        if profile_name:
            session_kwargs['profile_name'] = profile_name

        session = boto3.Session(**session_kwargs)
        self.bedrock_runtime = session.client(
            service_name='bedrock-runtime',
            region_name=os.environ.get('AWS_REGION', 'us-west-2')
        )

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate 1024-dimensional embedding using Titan V2.

        Args:
            text: Input text to embed

        Returns:
            List of 1024 floats representing the embedding vector

        Raises:
            ValueError: If embedding dimensions don't match expected size
            Exception: If Bedrock API call fails
        """
        try:
            request_body = {
                "inputText": text,
                "dimensions": 1024,
                "normalize": True
            }

            response = self.bedrock_runtime.invoke_model(
                modelId=self.TITAN_EMBED_MODEL,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )

            response_body = json.loads(response['body'].read())
            embedding = response_body['embedding']

            # Validate embedding dimensions
            if len(embedding) != self.EXPECTED_EMBEDDING_DIM:
                raise ValueError(
                    f"Expected {self.EXPECTED_EMBEDDING_DIM} dimensions, "
                    f"got {len(embedding)} dimensions from Titan"
                )

            return embedding

        except Exception as e:
            raise Exception(f"Failed to generate embedding: {str(e)}")

    def generate_recommendation(
        self,
        query: str,
        similar_experiments: List[Dict[str, Any]],
        relevant_memories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate recommendation using Claude Sonnet 4.5.

        Args:
            query: User's experiment description or query
            similar_experiments: List of similar past experiments
            relevant_memories: List of relevant agent memories

        Returns:
            Dict with structured recommendation data

        Raises:
            Exception: If Claude API call fails or response is invalid
        """
        try:
            # Build context from similar experiments
            experiments_context = self._format_experiments(similar_experiments)
            memories_context = self._format_memories(relevant_memories)

            # Construct prompt for Claude
            prompt = self._build_prompt(query, experiments_context, memories_context)

            # Invoke Claude Sonnet
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "temperature": 0.7,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }

            response = self.bedrock_runtime.invoke_model(
                modelId=self.CLAUDE_SONNET_MODEL,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )

            response_body = json.loads(response['body'].read())
            recommendation_text = response_body['content'][0]['text']

            # Strip markdown code fences if present
            # Claude sometimes wraps JSON in ```json ... ```
            if recommendation_text.strip().startswith('```'):
                lines = recommendation_text.strip().split('\n')
                # Remove first line (```json or ```)
                lines = lines[1:]
                # Remove last line if it's ```
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                recommendation_text = '\n'.join(lines).strip()

            # Parse JSON response
            try:
                recommendation_data = json.loads(recommendation_text)
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse Claude response as JSON: {type(e).__name__}")

            # Validate required fields
            required_fields = ['recommendation', 'confidence', 'suggested_approach',
                              'pitfalls', 'hyperparameters', 'expected_outcomes']
            missing_fields = [f for f in required_fields if f not in recommendation_data]
            if missing_fields:
                raise Exception(f"Missing required fields in response: {missing_fields}")

            # Validate confidence range
            confidence = recommendation_data['confidence']
            if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                raise Exception(f"Invalid confidence value: {confidence} (must be 0.0-1.0)")

            return recommendation_data

        except Exception as e:
            raise Exception(f"Failed to generate recommendation: {str(e)}")

    def _format_experiments(self, experiments: List[Dict[str, Any]]) -> str:
        """Format experiments for context."""
        if not experiments:
            return "No similar experiments found."

        formatted = []
        for i, exp in enumerate(experiments, 1):
            formatted.append(
                f"{i}. {exp['experiment_name']}\n"
                f"   Model: {exp.get('model_name', 'N/A')}\n"
                f"   Status: {exp['status']}\n"
                f"   Metrics: {json.dumps(exp.get('metrics', {}), indent=2)}\n"
                f"   Notes: {exp.get('notes', 'N/A')}"
            )

        return "\n\n".join(formatted)

    def _format_memories(self, memories: List[Dict[str, Any]]) -> str:
        """Format agent memories for context."""
        if not memories:
            return "No relevant memories found."

        formatted = []
        for i, mem in enumerate(memories, 1):
            formatted.append(
                f"{i}. [{mem['memory_type']}] {mem['content']}\n"
                f"   Context: {json.dumps(mem.get('context', {}))}"
            )

        return "\n\n".join(formatted)

    def _build_prompt(
        self,
        query: str,
        experiments_context: str,
        memories_context: str
    ) -> str:
        """Build prompt for Claude requesting JSON response."""
        return f"""You are an expert ML experiment advisor for ExperimentOps Agent.

User Query:
{query}

Similar Past Experiments:
{experiments_context}

Relevant Agent Memories:
{memories_context}

Based on the above context, provide a detailed recommendation for the user's experiment.

You MUST respond with valid JSON in this exact structure:
{{
  "recommendation": "Overall recommendation summary as a single paragraph",
  "confidence": 0.0,
  "suggested_approach": "Detailed approach based on past experiments",
  "pitfalls": ["Pitfall 1", "Pitfall 2", "..."],
  "hyperparameters": {{"param1": "value", "param2": "value"}},
  "expected_outcomes": "What the user should expect"
}}

Requirements:
- confidence must be a number between 0.0 and 1.0
- pitfalls must be an array of strings
- hyperparameters must be an object/dict
- All other fields must be strings
- Return ONLY the JSON, no other text"""
