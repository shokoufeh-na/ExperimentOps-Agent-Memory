"""
CockroachDB client for ExperimentOps Agent.
Handles all database operations including vector similarity search.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid


class CockroachDBClient:
    """Client for CockroachDB operations."""

    EXPECTED_EMBEDDING_DIM = 1024

    def __init__(self):
        """Initialize database connection from environment variables."""
        self.connection_string = os.environ.get('COCKROACHDB_URL')
        if not self.connection_string:
            raise ValueError("COCKROACHDB_URL environment variable is required")

        self.conn = None

    def connect(self):
        """
        Establish database connection.

        Raises:
            Exception: If connection fails (without exposing credentials)
        """
        try:
            if not self.conn or self.conn.closed:
                self.conn = psycopg2.connect(
                    self.connection_string,
                    cursor_factory=RealDictCursor
                )
        except Exception as e:
            raise Exception(f"Failed to connect to database: {type(e).__name__}")

    def close(self):
        """Close database connection."""
        if self.conn and not self.conn.closed:
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def insert_experiment(
        self,
        experiment_name: str,
        status: str,
        embedding: List[float],
        model_name: Optional[str] = None,
        parameters: Optional[Dict] = None,
        metrics: Optional[Dict] = None,
        notes: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> str:
        """
        Insert a new experiment into the database.

        Args:
            experiment_name: Name of the experiment
            status: 'running', 'completed', or 'failed'
            embedding: 1024-dimensional embedding vector
            model_name: ML model name
            parameters: Experiment parameters as dict
            metrics: Experiment metrics as dict
            notes: Additional notes
            error_message: Error message if failed

        Returns:
            UUID of the inserted experiment

        Raises:
            ValueError: If embedding dimensions are incorrect
            Exception: If database operation fails
        """
        # Validate embedding dimensions
        if len(embedding) != self.EXPECTED_EMBEDDING_DIM:
            raise ValueError(
                f"Expected {self.EXPECTED_EMBEDDING_DIM} dimensions, "
                f"got {len(embedding)} dimensions"
            )

        try:
            query = """
                INSERT INTO experiments (
                    experiment_name, model_name, parameters, metrics,
                    status, error_message, notes, embedding, embedding_model
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """

            with self.conn.cursor() as cur:
                cur.execute(query, (
                    experiment_name,
                    model_name,
                    Json(parameters) if parameters else None,
                    Json(metrics) if metrics else None,
                    status,
                    error_message,
                    notes,
                    str(embedding),  # CockroachDB VECTOR type accepts string representation
                    'amazon.titan-embed-text-v2:0'
                ))
                result = cur.fetchone()
                self.conn.commit()
                return str(result['id'])

        except Exception as e:
            self.conn.rollback()
            raise Exception(f"Failed to insert experiment: {type(e).__name__}")

    def search_similar_experiments(
        self,
        query_embedding: List[float],
        limit: int = 5,
        status_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find similar experiments using cosine similarity.

        Args:
            query_embedding: 1024-dimensional query embedding
            limit: Maximum number of results
            status_filter: Optional filter by status ('completed', 'failed', etc.)

        Returns:
            List of similar experiments with similarity scores

        Raises:
            ValueError: If embedding dimensions are incorrect
            Exception: If database operation fails
        """
        # Validate embedding dimensions
        if len(query_embedding) != self.EXPECTED_EMBEDDING_DIM:
            raise ValueError(
                f"Expected {self.EXPECTED_EMBEDDING_DIM} dimensions, "
                f"got {len(query_embedding)} dimensions"
            )

        try:
            # Build query with optional status filter
            embedding_str = str(query_embedding)

            if status_filter:
                query = """
                    SELECT
                        id,
                        experiment_name,
                        model_name,
                        parameters,
                        metrics,
                        status,
                        error_message,
                        notes,
                        created_at,
                        embedding <=> %s::vector AS distance
                    FROM experiments
                    WHERE embedding IS NOT NULL AND status = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """
                params = [embedding_str, status_filter, embedding_str, limit]
            else:
                query = """
                    SELECT
                        id,
                        experiment_name,
                        model_name,
                        parameters,
                        metrics,
                        status,
                        error_message,
                        notes,
                        created_at,
                        embedding <=> %s::vector AS distance
                    FROM experiments
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """
                params = [embedding_str, embedding_str, limit]

            with self.conn.cursor() as cur:
                cur.execute(query, params)
                results = cur.fetchall()

                # Convert to list of dicts and add similarity score
                experiments = []
                for row in results:
                    exp = dict(row)
                    exp['similarity'] = 1 - exp['distance']  # Convert distance to similarity
                    exp['id'] = str(exp['id'])  # Convert UUID to string
                    experiments.append(exp)

                return experiments

        except Exception as e:
            raise Exception(f"Failed to search experiments: {type(e).__name__}")

    def insert_agent_memory(
        self,
        memory_type: str,
        content: str,
        embedding: List[float],
        context: Optional[Dict] = None,
        relevance_score: Optional[float] = None
    ) -> str:
        """
        Insert a new agent memory.

        Args:
            memory_type: Type of memory (observation, insight, failure_pattern, success_pattern)
            content: Memory content
            embedding: 1024-dimensional embedding
            context: Additional context as dict
            relevance_score: Relevance score

        Returns:
            UUID of the inserted memory

        Raises:
            ValueError: If embedding dimensions are incorrect
            Exception: If database operation fails
        """
        # Validate embedding dimensions
        if len(embedding) != self.EXPECTED_EMBEDDING_DIM:
            raise ValueError(
                f"Expected {self.EXPECTED_EMBEDDING_DIM} dimensions, "
                f"got {len(embedding)} dimensions"
            )

        try:
            query = """
                INSERT INTO agent_memories (
                    memory_type, content, embedding, embedding_model,
                    context, relevance_score
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """

            with self.conn.cursor() as cur:
                cur.execute(query, (
                    memory_type,
                    content,
                    str(embedding),
                    'amazon.titan-embed-text-v2:0',
                    Json(context) if context else None,
                    relevance_score
                ))
                result = cur.fetchone()
                self.conn.commit()
                return str(result['id'])

        except Exception as e:
            self.conn.rollback()
            raise Exception(f"Failed to insert memory: {type(e).__name__}")

    def search_similar_memories(
        self,
        query_embedding: List[float],
        limit: int = 5,
        memory_type_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find similar agent memories using cosine similarity.

        Args:
            query_embedding: 1024-dimensional query embedding
            limit: Maximum number of results
            memory_type_filter: Optional filter by memory_type

        Returns:
            List of similar memories with similarity scores

        Raises:
            ValueError: If embedding dimensions are incorrect
            Exception: If database operation fails
        """
        # Validate embedding dimensions
        if len(query_embedding) != self.EXPECTED_EMBEDDING_DIM:
            raise ValueError(
                f"Expected {self.EXPECTED_EMBEDDING_DIM} dimensions, "
                f"got {len(query_embedding)} dimensions"
            )

        try:
            embedding_str = str(query_embedding)

            if memory_type_filter:
                query = """
                    SELECT
                        id,
                        memory_type,
                        content,
                        context,
                        relevance_score,
                        created_at,
                        accessed_at,
                        access_count,
                        embedding <=> %s::vector AS distance
                    FROM agent_memories
                    WHERE embedding IS NOT NULL AND memory_type = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """
                params = [embedding_str, memory_type_filter, embedding_str, limit]
            else:
                query = """
                    SELECT
                        id,
                        memory_type,
                        content,
                        context,
                        relevance_score,
                        created_at,
                        accessed_at,
                        access_count,
                        embedding <=> %s::vector AS distance
                    FROM agent_memories
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """
                params = [embedding_str, embedding_str, limit]

            with self.conn.cursor() as cur:
                cur.execute(query, params)
                results = cur.fetchall()

                memories = []
                for row in results:
                    mem = dict(row)
                    mem['similarity'] = 1 - mem['distance']
                    mem['id'] = str(mem['id'])
                    memories.append(mem)

                # Update access tracking
                memory_ids = [m['id'] for m in memories]
                self._update_memory_access(memory_ids)

                return memories

        except Exception as e:
            raise Exception(f"Failed to search memories: {type(e).__name__}")

    def _update_memory_access(self, memory_ids: List[str]):
        """Update access tracking for memories."""
        if not memory_ids:
            return

        try:
            query = """
                UPDATE agent_memories
                SET accessed_at = now(),
                    access_count = access_count + 1
                WHERE id = ANY(%s)
            """

            with self.conn.cursor() as cur:
                cur.execute(query, (memory_ids,))
                self.conn.commit()

        except Exception as e:
            self.conn.rollback()
            # Non-critical operation, log but don't raise
            print(f"Warning: Failed to update memory access: {type(e).__name__}")

    def insert_recommendation(
        self,
        experiment_id: Optional[str],
        recommendation_text: str,
        confidence: float,
        reasoning_model: str,
        applied: bool = False,
        outcome: Optional[Dict] = None
    ) -> str:
        """
        Insert a new recommendation.

        Args:
            experiment_id: Associated experiment UUID (optional)
            recommendation_text: The recommendation content
            confidence: Confidence score (0.0-1.0)
            reasoning_model: Model used for reasoning
            applied: Whether recommendation was applied
            outcome: Outcome data if applied

        Returns:
            UUID of the inserted recommendation

        Raises:
            Exception: If database operation fails
        """
        try:
            query = """
                INSERT INTO recommendations (
                    experiment_id, recommendation_text, confidence,
                    reasoning_model, applied, outcome
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """

            with self.conn.cursor() as cur:
                cur.execute(query, (
                    experiment_id if experiment_id else None,
                    recommendation_text,
                    confidence,
                    reasoning_model,
                    applied,
                    Json(outcome) if outcome else None
                ))
                result = cur.fetchone()
                self.conn.commit()
                return str(result['id'])

        except Exception as e:
            self.conn.rollback()
            raise Exception(f"Failed to insert recommendation: {type(e).__name__}")

    def get_experiment_by_id(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment by ID.

        Raises:
            Exception: If database operation fails
        """
        try:
            query = """
                SELECT id, experiment_name, model_name, parameters, metrics,
                       status, error_message, notes, created_at
                FROM experiments
                WHERE id = %s
            """

            with self.conn.cursor() as cur:
                cur.execute(query, (experiment_id,))
                result = cur.fetchone()
                if result:
                    exp = dict(result)
                    exp['id'] = str(exp['id'])
                    return exp
                return None

        except Exception as e:
            raise Exception(f"Failed to get experiment: {type(e).__name__}")
