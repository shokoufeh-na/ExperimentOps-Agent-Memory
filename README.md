ExperimentOps Agent
AI Agent with Persistent Memory using CockroachDB × AWS

Overview
ExperimentOps Agent is an AI agent designed for the CockroachDB × AWS Hackathon. It demonstrates how agentic applications can use CockroachDB as a durable, production‑grade memory layer and AWS for scalable execution. The agent stores experiment history, retrieves relevant past runs, analyzes failures, and recommends next steps — showing how persistent memory makes agents actually useful in real workflows.

What It Does
Tracks machine learning experiments and metadata

Stores long‑term agentic memory in CockroachDB

Uses distributed vector indexing for semantic search

Retrieves similar experiments and past failures

Uses Amazon Bedrock for reasoning and recommendations

Runs serverlessly on AWS Lambda
CockroachDB Tools Used
Managed MCP Server — agent‑native access to CockroachDB

Distributed Vector Indexing — semantic memory retrieval

Agent Skills Repo — optional DB optimization skills

AWS Services Used
AWS Lambda — agent execution

Amazon Bedrock — embeddings + reasoning

Optional: S3 for experiment artifacts

How It Works
The agent receives a query (e.g., “find experiments similar to a CUDA OOM failure”), generates an embedding, performs semantic search using CockroachDB’s vector index, analyzes retrieved experiments, and produces recommendations. Each interaction is stored as new memory, allowing the agent to improve over time.

Why This Project
This project highlights the core idea of the hackathon:
Agents need memory that never goes down.  
CockroachDB provides globally distributed, always‑on persistence, while AWS provides scalable execution and model intelligence. Together, they form a foundation for real agentic systems.
