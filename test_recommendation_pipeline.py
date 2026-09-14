#!/usr/bin/env python3
"""
Test complete recommendation pipeline with Claude Sonnet.
Generates structured recommendations based on retrieved experiments.
Read-only test - does not save recommendations to database yet.
"""

import sys
import os
import json

# Add lambda directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lambda'))

from bedrock_client import BedrockClient
from db import CockroachDBClient


def test_recommendation_pipeline():
    """Test end-to-end recommendation generation pipeline."""

    user_query = "I need to detect suspicious network traffic. Which previous experiment should I build on, and what should I try next?"

    print("=== Recommendation Pipeline Test ===")
    print()
    print(f"User Query: '{user_query}'")
    print()

    # Step 1: Initialize clients
    print("Step 1: Initialize clients")

    try:
        bedrock = BedrockClient(profile_name='experimentops')
        print("✓ BedrockClient initialized")
    except Exception as e:
        print(f"❌ BedrockClient initialization failed: {type(e).__name__}")
        return

    if not os.environ.get('COCKROACHDB_URL'):
        print("❌ COCKROACHDB_URL not set")
        return

    db_client = None
    try:
        db_client = CockroachDBClient()
        db_client.connect()
        print("✓ CockroachDBClient connected")
    except Exception as e:
        print(f"❌ Database connection failed: {type(e).__name__}")
        return

    print()

    # Step 2: Generate query embedding
    print("Step 2: Generate Titan embedding for query")

    try:
        query_embedding = bedrock.generate_embedding(user_query)
        print(f"✓ Query embedding generated ({len(query_embedding)} dimensions)")

        # Verify dimension
        if len(query_embedding) != 1024:
            print("❌ Invalid embedding dimension")
            if db_client:
                db_client.close()
            return

        print("✓ Query embedding verified: 1024 dimensions")

    except Exception as e:
        print(f"❌ Embedding generation failed: {type(e).__name__}")
        if db_client:
            db_client.close()
        return

    print()

    # Step 3: Retrieve similar experiments
    print("Step 3: Retrieve top 3 similar experiments")

    try:
        similar_experiments = db_client.search_similar_experiments(
            query_embedding=query_embedding,
            limit=3
        )

        print(f"✓ Retrieved {len(similar_experiments)} experiments")

        if len(similar_experiments) == 0:
            print("❌ No experiments found")
            if db_client:
                db_client.close()
            return

        print()
        print("Retrieved experiments:")
        for i, exp in enumerate(similar_experiments, 1):
            print(f"  {i}. {exp['experiment_name']}")
            print(f"     Model: {exp.get('model_name', 'N/A')}")
            print(f"     Similarity: {exp.get('similarity', 0):.4f}")

        # Verify Network Anomaly Detection is present (HARD REQUIREMENT)
        print()
        found_target = any(
            exp['experiment_name'] == "Network Anomaly Detection"
            for exp in similar_experiments
        )

        if not found_target:
            print("❌ TEST FAILED: 'Network Anomaly Detection' not found in top 3 results")
            if db_client:
                db_client.close()
            return

        print("✓ 'Network Anomaly Detection' found in results")

    except Exception as e:
        print(f"❌ Experiment retrieval failed: {type(e).__name__}")
        if db_client:
            db_client.close()
        return

    print()

    # Step 4: Generate Claude Sonnet recommendation
    print("Step 4: Generate recommendation using Claude Sonnet")

    try:
        # Pass empty list for agent memories (not testing those yet)
        recommendation = bedrock.generate_recommendation(
            query=user_query,
            similar_experiments=similar_experiments,
            relevant_memories=[]
        )

        print("✓ Recommendation generated")

    except Exception as e:
        print(f"❌ Recommendation generation failed: {type(e).__name__}")
        if db_client:
            db_client.close()
        return

    print()

    # Step 5: Validate recommendation structure
    print("Step 5: Validate recommendation structure")

    required_fields = [
        'recommendation',
        'confidence',
        'suggested_approach',
        'pitfalls',
        'hyperparameters',
        'expected_outcomes'
    ]

    all_fields_present = all(field in recommendation for field in required_fields)

    if all_fields_present:
        print("✓ All required fields present")
    else:
        missing = [f for f in required_fields if f not in recommendation]
        print(f"❌ Missing fields: {missing}")
        if db_client:
            db_client.close()
        return

    # Validate confidence
    confidence = recommendation.get('confidence', -1)
    if isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0:
        print(f"✓ Confidence valid: {confidence}")
    else:
        print(f"❌ Invalid confidence: {confidence}")
        if db_client:
            db_client.close()
        return

    # Validate JSON structure
    try:
        json_str = json.dumps(recommendation, indent=2)
        print("✓ Valid JSON structure")
    except Exception as e:
        print(f"❌ JSON serialization failed: {type(e).__name__}")
        if db_client:
            db_client.close()
        return

    print()

    # Step 6: Display recommendation (safe fields only)
    print("Step 6: Generated Recommendation")
    print("=" * 60)
    print()
    print(f"Recommendation Summary:")
    print(f"  {recommendation['recommendation']}")
    print()
    print(f"Confidence: {recommendation['confidence']}")
    print()
    print(f"Suggested Approach:")
    print(f"  {recommendation['suggested_approach']}")
    print()
    print(f"Pitfalls to Avoid:")
    for i, pitfall in enumerate(recommendation['pitfalls'], 1):
        print(f"  {i}. {pitfall}")
    print()
    print(f"Suggested Hyperparameters:")
    for key, value in recommendation['hyperparameters'].items():
        print(f"  {key}: {value}")
    print()
    print(f"Expected Outcomes:")
    print(f"  {recommendation['expected_outcomes']}")
    print()
    print("=" * 60)

    # Close connection
    if db_client:
        db_client.close()
        print()
        print("Step 7: Cleanup")
        print("✓ Database connection closed")

    print()
    print("=== Test Summary ===")
    print("✓ Query embedding: SUCCESS")
    print("✓ Experiment retrieval: SUCCESS")
    print("✓ Target experiment verification: SUCCESS")
    print("✓ Claude Sonnet recommendation: SUCCESS")
    print("✓ Validation: SUCCESS")
    print()
    print("✓ TEST PASSED")
    print()
    print("⚠ STOPPED: Recommendation NOT saved to database (as requested)")


if __name__ == '__main__':
    test_recommendation_pipeline()
