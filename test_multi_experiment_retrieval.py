#!/usr/bin/env python3
"""
Test multi-experiment semantic retrieval.
Inserts 4 new experiments and verifies semantic search ranking.
"""

import sys
import os

# Add lambda directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lambda'))

from bedrock_client import BedrockClient
from db import CockroachDBClient


def test_multi_experiment_retrieval():
    """Test semantic search ranking across multiple experiments."""

    # Define 4 new experiments
    experiments = [
        {
            "experiment_name": "Network Anomaly Detection",
            "model_name": "IsolationForest",
            "status": "completed",
            "notes": "Detects suspicious and anomalous network traffic for cybersecurity intrusion detection."
        },
        {
            "experiment_name": "CIFAR-10 Image Classification",
            "model_name": "CNN",
            "status": "completed",
            "notes": "Convolutional neural network for classifying images from the CIFAR-10 dataset."
        },
        {
            "experiment_name": "Sentiment Analysis",
            "model_name": "BERT",
            "status": "completed",
            "notes": "Transformer model for classifying positive and negative sentiment in text."
        },
        {
            "experiment_name": "Sales Forecasting",
            "model_name": "XGBoost",
            "status": "completed",
            "notes": "Predicts future product sales from historical business data."
        }
    ]

    print("=== Multi-Experiment Semantic Retrieval Test ===")
    print()

    # Initialize BedrockClient
    print("Step 1: Initialize BedrockClient")
    try:
        bedrock = BedrockClient(profile_name='experimentops')
        print("✓ BedrockClient initialized")
    except Exception as e:
        print(f"❌ BedrockClient initialization failed: {type(e).__name__}")
        return

    print()

    # Initialize CockroachDBClient
    print("Step 2: Connect to CockroachDB")

    if not os.environ.get('COCKROACHDB_URL'):
        print("❌ COCKROACHDB_URL not set")
        return

    db_client = None
    try:
        db_client = CockroachDBClient()
        db_client.connect()
        print("✓ CockroachDB connection established")
    except Exception as e:
        print(f"❌ Database connection failed: {type(e).__name__}")
        return

    print()

    # Insert experiments
    print("Step 3: Insert 4 new experiments")

    for i, exp in enumerate(experiments, 1):
        print(f"\n  Experiment {i}: {exp['experiment_name']}")

        try:
            # Check if experiment already exists
            with db_client.conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM experiments WHERE experiment_name = %s LIMIT 1",
                    (exp['experiment_name'],)
                )
                existing = cur.fetchone()

            if existing:
                print(f"    ⚠ Already exists (skipping)")
                continue

            # Generate embedding from name + notes
            text_for_embedding = f"{exp['experiment_name']}. {exp['notes']}"
            embedding = bedrock.generate_embedding(text_for_embedding)

            # Verify dimensions
            if len(embedding) != 1024:
                print(f"    ✗ Embedding dimension error: {len(embedding)} (expected 1024)")
                if db_client:
                    db_client.close()
                return

            print(f"    ✓ Embedding generated (1024 dimensions)")

            # Insert into database
            db_client.insert_experiment(
                experiment_name=exp['experiment_name'],
                status=exp['status'],
                embedding=embedding,
                model_name=exp['model_name'],
                notes=exp['notes']
            )

            print(f"    ✓ Inserted")

        except Exception as e:
            print(f"    ❌ Failed: {type(e).__name__}")
            if db_client:
                db_client.close()
            return

    print()
    print(f"✓ Experiment insertion phase completed")
    print()

    # Generate query embedding
    print("Step 4: Generate query embedding")
    query_text = "Which experiment can identify suspicious network activity and possible cyber attacks?"
    print(f"  Query: '{query_text}'")

    try:
        query_embedding = bedrock.generate_embedding(query_text)
        print("✓ Query embedding generated")
    except Exception as e:
        print(f"❌ Query embedding failed: {type(e).__name__}")
        if db_client:
            db_client.close()
        return

    print()

    # Search for similar experiments
    print("Step 5: Semantic search with limit=5")

    ranking_verified = False

    try:
        results = db_client.search_similar_experiments(
            query_embedding=query_embedding,
            limit=5
        )

        print(f"✓ Search completed")
        print(f"  Number of results: {len(results)}")
        print()

        if len(results) == 0:
            print("❌ No results returned")
        else:
            # Display results ranking
            print("Results (ranked by similarity):")
            print()

            for i, exp in enumerate(results, 1):
                print(f"  Rank {i}:")
                print(f"    Experiment: {exp['experiment_name']}")
                print(f"    Model: {exp.get('model_name', 'N/A')}")
                print(f"    Similarity: {exp.get('similarity', 0):.4f}")
                print()

            # Verify ranking
            if results[0]['experiment_name'] == "Network Anomaly Detection":
                ranking_verified = True

    except Exception as e:
        print(f"❌ Search failed: {type(e).__name__}")
    finally:
        if db_client:
            db_client.close()
            print("Step 6: Cleanup")
            print("✓ Database connection closed")
            print()

    # Final result
    print("=== Test Result ===")
    if ranking_verified:
        print("✓ TEST PASSED: 'Network Anomaly Detection' ranked #1")
    else:
        print("❌ TEST FAILED: 'Network Anomaly Detection' not ranked #1")

    print()
    print("Note: All test experiments remain in database")


if __name__ == '__main__':
    test_multi_experiment_retrieval()
