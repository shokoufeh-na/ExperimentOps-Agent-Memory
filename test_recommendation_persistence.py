#!/usr/bin/env python3
"""
Test recommendation persistence to CockroachDB.
Generates, inserts, and validates one recommendation record.
Uses [PERSISTENCE_TEST] marker for rerun safety.
"""

import sys
import os

# Add lambda directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lambda'))

from bedrock_client import BedrockClient
from db import CockroachDBClient


def test_recommendation_persistence():
    """Test writing and reading recommendation from database."""

    user_query = "I need to detect suspicious network traffic. Which previous experiment should I build on, and what should I try next?"
    test_marker = "[PERSISTENCE_TEST]"

    print("=== Recommendation Persistence Test ===")
    print()
    print(f"User Query: '{user_query}'")
    print(f"Test Marker: '{test_marker}'")
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

    # Step 2: Retrieve similar experiments
    print("Step 2: Retrieve top 3 similar experiments")

    try:
        query_embedding = bedrock.generate_embedding(user_query)
        print(f"✓ Query embedding generated ({len(query_embedding)} dimensions)")

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

        # Find Network Anomaly Detection experiment (HARD REQUIREMENT)
        target_experiment = None
        for exp in similar_experiments:
            if exp['experiment_name'] == "Network Anomaly Detection":
                target_experiment = exp
                break

        if not target_experiment:
            print("❌ TEST FAILED: 'Network Anomaly Detection' not found in top 3 results")
            if db_client:
                db_client.close()
            return

        print("✓ 'Network Anomaly Detection' found in results")
        target_experiment_id = target_experiment['id']
        print(f"  Target experiment selected")

    except Exception as e:
        print(f"❌ Experiment retrieval failed: {type(e).__name__}")
        if db_client:
            db_client.close()
        return

    print()

    # Step 3: Check for existing test recommendation
    print("Step 3: Check for existing test recommendation")

    try:
        with db_client.conn.cursor() as cur:
            cur.execute(
                """SELECT id FROM recommendations
                   WHERE experiment_id = %s
                   AND recommendation_text LIKE %s
                   LIMIT 1""",
                (target_experiment_id, f"{test_marker}%")
            )
            existing = cur.fetchone()

        if existing:
            print(f"⚠ Test recommendation already exists (rerun verification)")
            existing_recommendation_id = existing['id']
            skip_insert = True
        else:
            print("✓ No test recommendation found (will create new)")
            skip_insert = False

    except Exception as e:
        print(f"❌ Existing recommendation check failed: {type(e).__name__}")
        if db_client:
            db_client.close()
        return

    print()

    # Step 4: Generate recommendation (if needed)
    if not skip_insert:
        print("Step 4: Generate Claude Sonnet recommendation")

        try:
            recommendation_data = bedrock.generate_recommendation(
                query=user_query,
                similar_experiments=similar_experiments,
                relevant_memories=[]
            )

            print("✓ Recommendation generated")
            print(f"  Confidence: {recommendation_data['confidence']}")

        except Exception as e:
            print(f"❌ Recommendation generation failed: {type(e).__name__}")
            if db_client:
                db_client.close()
            return

        print()

        # Step 5: Insert recommendation with test marker
        print("Step 5: Insert recommendation into database")

        try:
            # Prepend test marker to recommendation text
            marked_recommendation_text = f"{test_marker} {recommendation_data['recommendation']}"

            recommendation_id = db_client.insert_recommendation(
                experiment_id=target_experiment_id,
                recommendation_text=marked_recommendation_text,
                confidence=recommendation_data['confidence'],
                reasoning_model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                applied=False
            )

            print("✓ Recommendation inserted")
            print(f"  New test record created")

        except Exception as e:
            print(f"❌ Recommendation insertion failed: {type(e).__name__}")
            if db_client:
                db_client.close()
            return

        print()
    else:
        print("Step 4-5: Skipped (using existing test recommendation)")
        recommendation_id = existing_recommendation_id
        print()

    # Step 6: Read back recommendation by exact ID
    print("Step 6: Read recommendation from database by ID")

    try:
        with db_client.conn.cursor() as cur:
            cur.execute(
                """SELECT experiment_id, recommendation_text, confidence,
                          reasoning_model, applied
                   FROM recommendations
                   WHERE id = %s""",
                (recommendation_id,)
            )
            recommendation = cur.fetchone()

        if not recommendation:
            print("❌ Failed to read recommendation by ID")
            if db_client:
                db_client.close()
            return

        print("✓ Recommendation read from database")

    except Exception as e:
        print(f"❌ Recommendation read failed: {type(e).__name__}")
        if db_client:
            db_client.close()
        return

    print()

    # Step 7: Validate retrieved data
    print("Step 7: Validate recommendation data")

    validation_passed = True

    # Validate experiment_id
    if recommendation['experiment_id'] == target_experiment_id:
        print("✓ experiment_id matches")
    else:
        print("❌ experiment_id mismatch")
        validation_passed = False

    # Validate recommendation_text starts with test marker
    if recommendation['recommendation_text'].startswith(test_marker):
        print(f"✓ recommendation_text begins with {test_marker}")
        text_length = len(recommendation['recommendation_text'])
        print(f"✓ recommendation_text is non-empty ({text_length} chars)")
    else:
        print(f"❌ recommendation_text does not begin with {test_marker}")
        validation_passed = False

    # Validate confidence
    confidence = recommendation['confidence']
    if isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0:
        print(f"✓ confidence is valid: {confidence}")
    else:
        print(f"❌ confidence is invalid: {confidence}")
        validation_passed = False

    # Validate reasoning_model
    expected_model = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    if recommendation['reasoning_model'] == expected_model:
        print(f"✓ reasoning_model matches")
    else:
        print(f"❌ reasoning_model mismatch: {recommendation['reasoning_model']}")
        validation_passed = False

    # Validate applied
    if recommendation['applied'] is False:
        print("✓ applied is False")
    else:
        print(f"❌ applied is not False: {recommendation['applied']}")
        validation_passed = False

    # Close connection
    if db_client:
        db_client.close()
        print()
        print("Step 8: Cleanup")
        print("✓ Database connection closed")

    print()
    print("=" * 60)
    print("=== Test Summary ===")
    print()

    if not skip_insert:
        print("Test Type: NEW INSERTION")
    else:
        print("Test Type: RERUN VERIFICATION")

    print()

    if validation_passed:
        print("✓ All validations PASSED")
        print()
        print("✓ TEST PASSED")
        print()
        print("Recommendation successfully persisted and verified in CockroachDB")
    else:
        print("❌ Some validations FAILED")
        print()
        print("❌ TEST FAILED")


if __name__ == '__main__':
    test_recommendation_persistence()
