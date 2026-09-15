# ExperimentOps Agent

**AI Agent with Persistent Memory using CockroachDB × AWS**

ExperimentOps Agent is an AI-powered system built for the **CockroachDB × AWS Hackathon**. It gives an AI agent durable memory of previous machine-learning experiments so it can retrieve similar runs, recall past failures and insights, and recommend better next steps.

Instead of treating every query as a new interaction, ExperimentOps Agent uses **CockroachDB as persistent agent memory** and **AWS Bedrock** for embeddings and reasoning.

## What It Does

* Tracks machine-learning experiments, parameters, metrics, status, and failures
* Stores persistent agent memories and experiment insights in CockroachDB
* Generates **1024-dimensional embeddings** using Amazon Titan Embed Text v2
* Uses CockroachDB vector indexes for semantic similarity search
* Retrieves relevant experiments and previously learned insights
* Uses Claude Sonnet on Amazon Bedrock to generate recommendations
* Stores generated recommendations for future use
* Runs as a containerized serverless application on AWS Lambda

## Architecture

```text
User Query
    ↓
AWS Lambda
    ↓
Amazon Titan Embed Text v2
    ↓
1024-D Query Embedding
    ↓
CockroachDB Vector Search
    ├── Similar Experiments
    └── Relevant Agent Memories
    ↓
Claude Sonnet
    ↓
Structured Recommendation
    ↓
CockroachDB Persistent Memory
```

## Example

A user can ask:

> I want to improve network anomaly detection while reducing false positives. What previous experiment and stored memory should I use?

The deployed agent:

1. Generates an embedding for the query.
2. Searches CockroachDB for semantically similar experiments.
3. Retrieves relevant persistent agent memories.
4. Sends the retrieved context to Claude Sonnet.
5. Generates a structured recommendation with suggested approaches, hyperparameters, pitfalls, and expected outcomes.
6. Stores the recommendation in CockroachDB for future use.

In an end-to-end deployment test, the system retrieved **5 related experiments**, identified a relevant network-anomaly experiment and stored insight, and generated a recommendation with **0.85 confidence**.

## CockroachDB

CockroachDB serves as the durable memory and retrieval layer.

### Tables

* `experiments` — experiment metadata, parameters, metrics, status, and embeddings
* `agent_memories` — persistent insights and agent knowledge
* `recommendations` — AI-generated recommendations and outcomes

### Vector Search

Experiment and memory embeddings use:

```text
VECTOR(1024)
```

with cosine similarity search through CockroachDB vector indexes.

The system can therefore retrieve experiments based on semantic similarity rather than relying only on keywords.

## AWS Services

### AWS Lambda

Runs the ExperimentOps agent as a serverless containerized application.

### Amazon Bedrock

**Titan Embed Text v2**

```text
amazon.titan-embed-text-v2:0
```

Generates 1024-dimensional embeddings for semantic retrieval.

**Claude Sonnet**

Used to analyze retrieved experiment context and generate structured recommendations.

### Amazon ECR

Stores the Docker container image used by AWS Lambda.

## Persistent Agent Memory

The agent does more than retrieve experiment records.

Important observations can be stored as semantic memories, for example:

```text
Network anomaly experiments performed better when contamination
was tuned carefully and false-positive rate was evaluated
alongside recall.
```

Future queries can retrieve these memories through vector similarity search, allowing previous experiment knowledge to influence later recommendations.

## Security

Database connections between AWS Lambda and CockroachDB use TLS certificate verification:

```text
sslmode=verify-full
```

The trusted CA certificate is bundled into the Lambda container and used for full server certificate verification.

Credentials and database connection strings are not stored in the repository.

## Deployment

The application is containerized using the AWS Lambda Python 3.12 base image.

```text
Docker
   ↓
Amazon ECR
   ↓
AWS Lambda
   ↓
Amazon Bedrock
   ↓
CockroachDB Cloud
```

The Lambda execution role uses scoped IAM permissions for:

* Amazon Titan embedding inference
* Claude Sonnet inference
* Bedrock inference-profile routing
* CloudWatch logging

## Repository Structure

```text
ExperimentOps-Agent-Memory/
├── lambda/
│   ├── handler.py
│   ├── bedrock_client.py
│   ├── db.py
│   ├── requirements.txt
│   └── certs/
│       └── root.crt
│
├── sql/
│   ├── schema.sql
│   └── seed_data.sql
│
├── tests/
│   ├── test_embeddings.py
│   └── test_retrieval.py
│
├── mcp/
│   └── README.md
│
├── diagrams/
├── Dockerfile
├── .env.example
├── .gitignore
└── README.md
```

## Local Configuration

Create the required environment variables locally. Never commit credentials to Git.

Example:

```text
COCKROACHDB_URL=<your-secure-connection-string>
AWS_REGION=us-west-2
```

## Current Status

The complete deployed pipeline has been tested successfully:

```text
AWS Lambda                     ✓
Amazon Titan embeddings        ✓
CockroachDB TLS connection     ✓
Vector experiment retrieval    ✓
Persistent memory retrieval    ✓
Claude Sonnet reasoning        ✓
Recommendation generation      ✓
Persistent recommendation      ✓
```

## Why ExperimentOps Agent?

ML experimentation produces valuable information about what worked, what failed, and why. That knowledge is often scattered across logs, notebooks, experiment trackers, and individual developers.

ExperimentOps Agent turns previous experiments into **retrievable AI memory**.

CockroachDB provides durable and scalable storage and vector retrieval, while AWS provides serverless execution and foundation-model intelligence. Together they allow the agent to remember previous work and use that knowledge when making future recommendations.

## Built For

**CockroachDB × AWS Hackathon — Agentic Memory**
