#!/usr/bin/env python3
"""
Integration test for lambda/handler.py.
Tests the complete query/recommendation flow with real Bedrock and CockroachDB.
"""

import sys
import os
import json

# Add lambda directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lambda'))

# Import handler
from handler import handler


def test_handler_integration():
    """Test handler query flow with real services."""

    print("=== Lambda Handler Integration Test ===")
    print()
    print("Testing action: query")
    print("Using: Real Bedrock + Real CockroachDB")
    print()

    # Verify environment
    print("Step 1: Verify environment")

    if not os.environ.get('COCKROACHDB_URL'):
        print("❌ COCKROACHDB_URL not set")
        return
    print("✓ COCKROACHDB_URL is set")

    # Set LOCAL_AWS_PROFILE for local testing
    os.environ['LOCAL_AWS_PROFILE'] = 'experimentops'
    print("✓ LOCAL_AWS_PROFILE set to: experimentops")

    print()

    # Build test event
    print("Step 2: Build test event")

    test_query = (
        "I want to improve network anomaly detection while reducing false positives. "
        "What previous experiment and stored memory should I use?"
    )

    test_event = {
        "action": "query",
        "query": test_query
    }

    print(f"Query: '{test_query}'")
    print()

    # Expected database writes
    print("Expected Database Writes:")
    print("  1. insert_recommendation() - 1 new row")
    print("  2. search_similar_memories() - updates accessed_at/access_count for retrieved memories")
    print()

    # Call handler
    print("Step 3: Call lambda_handler()")

    try:
        response = handler(test_event, None)
        print("✓ Handler returned")
    except Exception as e:
        print(f"❌ Handler failed: {type(e).__name__}")
        return

    print()

    # Validate response structure
    print("Step 4: Validate response structure")

    validation_passed = True

    # Check response is dict
    if isinstance(response, dict):
        print("✓ Response is dict")
    else:
        print(f"❌ Response is not dict: {type(response).__name__}")
        validation_passed = False
        return

    # Check statusCode
    status_code = response.get('statusCode')
    if status_code == 200:
        print(f"✓ statusCode is 200")
    else:
        print(f"❌ statusCode is {status_code} (expected 200)")
        validation_passed = False

    # Check body exists
    body_str = response.get('body')
    if body_str:
        print("✓ body field exists")
    else:
        print("❌ body field missing")
        validation_passed = False
        return

    # Parse body JSON
    try:
        body = json.loads(body_str)
        print("✓ body is valid JSON")
    except json.JSONDecodeError as e:
        print(f"❌ body is not valid JSON: {type(e).__name__}")
        validation_passed = False
        return

    print()

    # Validate response content
    print("Step 5: Validate response content")

    # Check required fields
    required_fields = [
        'recommendation',
        'confidence',
        'suggested_approach',
        'pitfalls',
        'hyperparameters',
        'expected_outcomes',
        'recommendation_id',
        'similar_experiments_count',
        'relevant_memories_count',
        'context'
    ]

    for field in required_fields:
        if field in body:
            print(f"✓ {field} present")
        else:
            print(f"❌ {field} missing")
            validation_passed = False

    print()

    # Validate specific fields
    print("Step 6: Validate field values")

    # Recommendation text
    recommendation = body.get('recommendation', '')
    if recommendation and len(recommendation) > 0:
        print(f"✓ recommendation is non-empty ({len(recommendation)} chars)")
    else:
        print("❌ recommendation is empty")
        validation_passed = False

    # Confidence
    confidence = body.get('confidence')
    if isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0:
        print(f"✓ confidence is valid: {confidence}")
    else:
        print(f"❌ confidence is invalid: {confidence}")
        validation_passed = False

    # Similar experiments count consistency
    exp_count = body.get('similar_experiments_count', 0)
    context = body.get('context', {})
    similar_experiments = context.get('similar_experiments', [])

    if exp_count == len(similar_experiments):
        print(f"✓ similar_experiments_count matches context: {exp_count}")
    else:
        print(f"❌ Count mismatch: similar_experiments_count={exp_count}, context has {len(similar_experiments)}")
        validation_passed = False

    # Relevant memories count consistency
    mem_count = body.get('relevant_memories_count', 0)
    relevant_memories = context.get('relevant_memories', [])

    if mem_count == len(relevant_memories):
        print(f"✓ relevant_memories_count matches context: {mem_count}")
    else:
        print(f"❌ Count mismatch: relevant_memories_count={mem_count}, context has {len(relevant_memories)}")
        validation_passed = False

    # Recommendation ID
    recommendation_id = body.get('recommendation_id')
    if recommendation_id and len(str(recommendation_id)) > 0:
        print("✓ recommendation_id is present and non-empty")
    else:
        print("❌ recommendation_id is missing or empty")
        validation_passed = False

    print()

    # Validate context
    print("Step 7: Validate context data")

    # Similar experiments (HARD REQUIREMENT)
    print(f"Similar experiments returned: {len(similar_experiments)}")

    if len(similar_experiments) == 0:
        print("❌ HARD REQUIREMENT FAILED: No similar experiments returned")
        validation_passed = False
    else:
        print("✓ At least one similar experiment returned")

        # Check if Network Anomaly Detection is included (HARD REQUIREMENT)
        network_anomaly_found = any(
            'network anomaly' in exp.get('name', '').lower()
            for exp in similar_experiments
        )

        if network_anomaly_found:
            print("✓ 'Network Anomaly Detection' found in similar experiments")
        else:
            print("❌ HARD REQUIREMENT FAILED: 'Network Anomaly Detection' not found")
            print("  Returned experiments:")
            for i, exp in enumerate(similar_experiments, 1):
                print(f"    {i}. {exp.get('name', 'N/A')} (similarity: {exp.get('similarity', 0):.4f})")
            validation_passed = False

    print()

    # Relevant memories (HARD REQUIREMENT)
    print(f"Relevant memories returned: {len(relevant_memories)}")

    if len(relevant_memories) == 0:
        print("❌ HARD REQUIREMENT FAILED: No relevant memories returned")
        validation_passed = False
    else:
        print("✓ At least one relevant memory returned")

        # Validate each memory has required fields
        all_memories_valid = True
        for i, mem in enumerate(relevant_memories, 1):
            mem_type = mem.get('type', '')
            mem_similarity = mem.get('similarity', None)

            # Validate type
            if not mem_type or len(mem_type) == 0:
                print(f"  ❌ Memory {i}: type is empty")
                all_memories_valid = False

            # Validate similarity
            if not isinstance(mem_similarity, (int, float)):
                print(f"  ❌ Memory {i}: similarity is not numeric")
                all_memories_valid = False

        if all_memories_valid:
            print("✓ All returned memories have valid type and similarity")
            print("  Memory details:")
            for i, mem in enumerate(relevant_memories, 1):
                print(f"    {i}. Type: {mem.get('type', 'N/A')} (similarity: {mem.get('similarity', 0):.4f})")
        else:
            validation_passed = False

    print()

    # Check for credential leakage
    print("Step 8: Security validation")

    body_str_lower = body_str.lower()

    unsafe_patterns = [
        'cockroachdb_url',
        'aws_access_key',
        'aws_secret',
        'password',
        'traceback',
        'exception in'
    ]

    found_unsafe = []
    for pattern in unsafe_patterns:
        if pattern in body_str_lower:
            found_unsafe.append(pattern)

    if found_unsafe:
        print(f"❌ Found unsafe content: {found_unsafe}")
        validation_passed = False
    else:
        print("✓ No credentials or internal errors in response")

    print()

    # Final summary
    print("=" * 60)
    print("=== Test Summary ===")
    print()

    print("Database Writes Performed:")
    print(f"  - 1 new recommendation row")
    print(f"  - {mem_count} memory access updates (accessed_at/access_count)")
    print()

    if validation_passed:
        print("✓ All validations PASSED")
        print()
        print("✓ TEST PASSED")
        print()
        print("Handler integration verified with real Bedrock and CockroachDB")
    else:
        print("❌ Some validations FAILED")
        print()
        print("❌ TEST FAILED")


if __name__ == '__main__':
    test_handler_integration()
