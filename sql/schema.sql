-- ExperimentOps Agent Memory Schema
-- CockroachDB v26.2.1 compatible schema for experiment tracking and agent memory

-- Experiments table: tracks ML experiments with metadata
CREATE TABLE experiments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_name STRING NOT NULL,
    model_name STRING,
    parameters JSONB,
    metrics JSONB,
    status STRING NOT NULL, -- 'running', 'completed', 'failed'
    error_message STRING,
    notes STRING,
    embedding VECTOR(1024), -- Amazon Titan embedding dimension
    embedding_model STRING, -- e.g., 'amazon.titan-embed-text-v2:0'
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Agent memories table: stores long-term agent memory and context
CREATE TABLE agent_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_type STRING NOT NULL, -- 'observation', 'insight', 'failure_pattern', 'success_pattern'
    content STRING NOT NULL,
    embedding VECTOR(1024),
    embedding_model STRING, -- e.g., 'amazon.titan-embed-text-v2:0'
    context JSONB,
    relevance_score FLOAT8,
    created_at TIMESTAMPTZ DEFAULT now(),
    accessed_at TIMESTAMPTZ DEFAULT now(),
    access_count INT DEFAULT 0
);

-- Recommendations table: agent-generated recommendations
CREATE TABLE recommendations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID REFERENCES experiments(id) ON DELETE CASCADE,
    recommendation_text STRING NOT NULL,
    confidence FLOAT8,
    reasoning_model STRING, -- e.g., 'us.anthropic.claude-sonnet-4-5-20250929-v1:0'
    applied BOOLEAN DEFAULT false,
    outcome JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Enable vector indexes (required for CockroachDB v26.2.1)
SET CLUSTER SETTING feature.vector_index.enabled = true;

-- CockroachDB native VECTOR indexes for semantic search
CREATE VECTOR INDEX experiments_embedding_idx
    ON experiments (embedding vector_cosine_ops);

CREATE VECTOR INDEX agent_memories_embedding_idx
    ON agent_memories (embedding vector_cosine_ops);

-- Standard indexes for common queries
CREATE INDEX experiments_status_idx ON experiments(status);
CREATE INDEX experiments_created_at_idx ON experiments(created_at DESC);

CREATE INDEX agent_memories_type_idx ON agent_memories(memory_type);
CREATE INDEX agent_memories_accessed_at_idx ON agent_memories(accessed_at DESC);

CREATE INDEX recommendations_experiment_id_idx ON recommendations(experiment_id);
CREATE INDEX recommendations_applied_idx ON recommendations(applied);
