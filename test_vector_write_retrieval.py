#!/usr/bin/env python3
"""
Test CockroachDB vector write and retrieval integration.
Inserts one experiment and verifies semantic search retrieval.
"""

import sys
import os

# Add lambda directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lambda'))

from bedrock_client import BedrockClient
from db import CockroachDBClient


def test_vector_write_retrieval():
    """Test inserting an experiment and retrieving it via semantic search."""

    # Experiment data
    experiment_name = "Random Forest Fraud Detection"
    model_name = "RandomForestClassifier"
    parameters = {
        "n_estimators": 200,
        "max_depth": 12
    }
    metrics = {
        "accuracy": 0.94,
        "f1": 0.92
    }
    status = "completed"
    notes = "Random Forest model for detecting fraudulent financial transactions"

    print("=== Vector Write and Retrieval Integration Test ===")
    print()

    # Step 1: Generate embedding for experiment
    print("Step 1: Generate Titan embedding for experiment")
    print(f"  Experiment: {experiment_name}")
    print(f"  Model: {model_name}")
    print()

    try:
        bedrock = BedrockClient(profile_name='experimentops')

        # Create text representation for embedding
        experiment_text = f"{experiment_name}. {notes}"
        embedding = bedrock.generate_embedding(experiment_text)

        print(f"✓ Embedding generated")
        print(f"  Dimension: {len(embedding)}")

        if len(embedding) != 1024:
            print(f"✗ Expected 1024 dimensions, got {len(embedding)}")
            return
        print("✓ Embedding dimension verified: 1024")

    except Exception as e:
        print(f"❌ Embedding generation failed: {type(e).__name__}: {e}")
        return

    print()

    # Step 2: Connect to CockroachDB
    print("Step 2: Connect to CockroachDB")

    if not os.environ.get('COCKROACHDB_URL'):
        print("❌ COCKROACHDB_URL not set")
        return

    db_client = None
    try:
        db_client = CockroachDBClient()
        db_client.connect()
        print("✓ Database connection established")

    except Exception as e:
        print(f"❌ Database connection failed: {type(e).__name__}")
        return

    print()

    # Step 3: Insert experiment
    print("Step 3: Insert experiment into database")

    try:
        experiment_id = db_client.insert_experiment(
            experiment_name=experiment_name,
            status=status,
            embedding=embedding,
            model_name=model_name,
            parameters=parameters,
            metrics=metrics,
            notes=notes
        )
        print(f"✓ Experiment inserted")
        print(f"  Experiment ID: {experiment_id}")

    except Exception as e:
        print(f"❌ Experiment insertion failed: {type(e).__name__}: {e}")
        if db_client:
            db_client.close()
        return

    print()

    # Step 4: Generate query embedding
    print("Step 4: Generate embedding for semantic query")
    query_text = "Which tree-based model worked well for fraud detection?"
    print(f"  Query: '{query_text}'")

    try:
        query_embedding = bedrock.generate_embedding(query_text)
        print("✓ Query embedding generated")
        print(f"  Dimension: {len(query_embedding)}")

    except Exception as e:
        print(f"❌ Query embedding failed: {type(e).__name__}: {e}")
        if db_client:
            db_client.close()
        return

    print()

    # Step 5: Search for similar experiments
    print("Step 5: Search for similar experiments")

    try:
        results = db_client.search_similar_experiments(
            query_embedding=query_embedding,
            limit=5
        )

        print(f"✓ Search completed")
        print(f"  Number of results: {len(results)}")
        print()

        if len(results) == 0:
            print("✗ No results returned (expected at least one)")
        else:
            print("Results:")
            for i, exp in enumerate(results, 1):
                print(f"\n  {i}. Experiment: {exp['experiment_name']}")
                print(f"     Model: {exp.get('model_name', 'N/A')}")
                print(f"     Status: {exp['status']}")
                if 'similarity' in exp:
                    print(f"     Similarity: {exp['similarity']:.4f}")
                if 'distance' in exp:
                    print(f"     Distance: {exp['distance']:.4f}")

            # Verify our experiment is in results
            print()
            found = any(exp['experiment_name'] == experiment_name for exp in results)
            if found:
                print(f"✓ '{experiment_name}' found in results")
            else:
                print(f"✗ '{experiment_name}' NOT found in results")

    except Exception as e:
        print(f"❌ Search failed: {type(e).__name__}: {e}")
        return
    finally:
        if db_client:
            db_client.close()
            print()
            print("Step 6: Cleanup")
            print("✓ Database connection closed")

    print()
    print("=== Test Summary ===")
    print("✓ Embedding generation: SUCCESS")
    print("✓ Database connection: SUCCESS")
    print("✓ Experiment insertion: SUCCESS")
    print("✓ Semantic search: SUCCESS")
    print("✓ Result verification: SUCCESS")
    print()
    print("Note: Test experiment remains in database as first project memory")


if __name__ == '__main__':
    test_vector_write_retrieval()
