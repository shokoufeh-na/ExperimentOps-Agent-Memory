#!/usr/bin/env python3
"""
Test agent memory persistence and semantic retrieval.
Validates storing and retrieving memories with Titan embeddings.
Uses [MEMORY_TEST] marker for rerun safety.
"""

import sys
import os

# Add lambda directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lambda'))

from bedrock_client import BedrockClient
from db import CockroachDBClient


def test_agent_memory():
    """Test agent memory storage and retrieval pipeline."""

    test_marker = "[MEMORY_TEST]"
    test_memory = (
        "[MEMORY_TEST] Network anomaly experiments performed better when "
        "contamination was tuned carefully and false-positive rate was "
        "evaluated alongside recall."
    )

    semantic_query = (
        "What did we learn about reducing false positives "
        "in network anomaly detection?"
    )

    print("=== Agent Memory Pipeline Test ===")
    print()
    print(f"Test Marker: '{test_marker}'")
    print(f"Test Memory: '{test_memory}'")
    print()
    print(f"NOTE: search_similar_memories() performs database writes:")
    print(f"  - Updates accessed_at timestamp")
    print(f"  - Increments access_count")
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

    # Step 2: Generate embedding for test memory
    print("Step 2: Generate Titan embedding for test memory")

    try:
        memory_embedding = bedrock.generate_embedding(test_memory)
        print(f"✓ Memory embedding generated ({len(memory_embedding)} dimensions)")

        # HARD REQUIREMENT: Verify dimension
        if len(memory_embedding) != 1024:
            print(f"❌ FAILED: Expected 1024 dimensions, got {len(memory_embedding)}")
            if db_client:
                db_client.close()
            return

        print("✓ Embedding dimension verified: 1024")

    except Exception as e:
        print(f"❌ Embedding generation failed: {type(e).__name__}")
        if db_client:
            db_client.close()
        return

    print()

    # Step 3: Check for existing test memory
    print("Step 3: Check for existing test memory")

    try:
        with db_client.conn.cursor() as cur:
            cur.execute(
                """SELECT id FROM agent_memories
                   WHERE content LIKE %s
                   LIMIT 1""",
                (f"{test_marker}%",)
            )
            existing = cur.fetchone()

        if existing:
            print(f"⚠ Test memory already exists (rerun verification)")
            existing_memory_id = existing['id']
            skip_insert = True
        else:
            print("✓ No test memory found (will create new)")
            skip_insert = False

    except Exception as e:
        print(f"❌ Existing memory check failed: {type(e).__name__}")
        if db_client:
            db_client.close()
        return

    print()

    # Step 4: Insert memory (if needed)
    if not skip_insert:
        print("Step 4: Insert agent memory")

        try:
            memory_id = db_client.insert_agent_memory(
                memory_type="experiment_insight",
                content=test_memory,
                embedding=memory_embedding,
                context={"source": "memory_pipeline_test"}
            )

            print("✓ Agent memory inserted")
            print(f"  Memory type: experiment_insight")
            print(f"  Test record created")
            print(f"  Note: embedding_model automatically set to amazon.titan-embed-text-v2:0")

        except Exception as e:
            print(f"❌ Memory insertion failed: {type(e).__name__}")
            if db_client:
                db_client.close()
            return

        print()
    else:
        print("Step 4: Skipped (using existing test memory)")
        memory_id = existing_memory_id
        print()

    # Step 5: Generate embedding for semantic query
    print("Step 5: Generate embedding for semantic query")
    print(f"  Query: '{semantic_query}'")

    try:
        query_embedding = bedrock.generate_embedding(semantic_query)
        print(f"✓ Query embedding generated ({len(query_embedding)} dimensions)")

        # HARD REQUIREMENT: Verify dimension
        if len(query_embedding) != 1024:
            print(f"❌ FAILED: Query embedding must be 1024 dimensions, got {len(query_embedding)}")
            if db_client:
                db_client.close()
            return

        print("✓ Query embedding dimension verified: 1024")

    except Exception as e:
        print(f"❌ Query embedding failed: {type(e).__name__}")
        if db_client:
            db_client.close()
        return

    print()

    # Step 6: Search similar memories
    print("Step 6: Search similar agent memories")
    print("  Note: This will update accessed_at and access_count")

    try:
        similar_memories = db_client.search_similar_memories(
            query_embedding=query_embedding,
            limit=5
        )

        print(f"✓ Search completed")
        print(f"  Retrieved {len(similar_memories)} memories")

        if len(similar_memories) == 0:
            print("❌ No memories retrieved")
            if db_client:
                db_client.close()
            return

        print()
        print("Retrieved memories:")
        for i, mem in enumerate(similar_memories, 1):
            print(f"  {i}. Type: {mem['memory_type']}")
            print(f"     Similarity: {mem.get('similarity', 0):.4f}")
            if mem['content'].startswith(test_marker):
                print(f"     Content: {test_marker} (test memory)")
            else:
                print(f"     Content: {mem['content'][:60]}...")

        print()

        # HARD REQUIREMENT: Test memory must be in results
        found_test_memory = any(
            mem['content'].startswith(test_marker)
            for mem in similar_memories
        )

        if not found_test_memory:
            print(f"❌ TEST FAILED: {test_marker} memory not found in results")
            if db_client:
                db_client.close()
            return

        print(f"✓ {test_marker} memory found in results")

        # Get the test memory for validation
        test_memory_result = next(
            mem for mem in similar_memories
            if mem['content'].startswith(test_marker)
        )

    except Exception as e:
        print(f"❌ Memory search failed: {type(e).__name__}")
        if db_client:
            db_client.close()
        return

    print()

    # Step 7: Validate retrieved memory
    print("Step 7: Validate retrieved memory")

    validation_passed = True

    # Validate memory_type
    if test_memory_result['memory_type'] == "experiment_insight":
        print("✓ memory_type is 'experiment_insight'")
    else:
        print(f"❌ memory_type mismatch: {test_memory_result['memory_type']}")
        validation_passed = False

    # Validate content starts with marker
    if test_memory_result['content'].startswith(test_marker):
        print(f"✓ content begins with {test_marker}")
    else:
        print(f"❌ content does not begin with {test_marker}")
        validation_passed = False

    # Validate similarity is numeric
    similarity = test_memory_result.get('similarity', None)
    if isinstance(similarity, (int, float)):
        print(f"✓ similarity is valid numeric: {similarity:.4f}")
    else:
        print(f"❌ similarity is not numeric: {similarity}")
        validation_passed = False

    # Validate context
    context = test_memory_result.get('context', {})
    if context and context.get('source') == 'memory_pipeline_test':
        print(f"✓ context contains test source marker")
    else:
        print(f"⚠ context may not match (expected after rerun)")

    print()

    # Step 8: Validate embedding_model from database
    print("Step 8: Validate embedding_model")
    print("  Note: embedding_model not returned by search, reading by ID")

    try:
        with db_client.conn.cursor() as cur:
            cur.execute(
                """SELECT embedding_model FROM agent_memories
                   WHERE id = %s""",
                (test_memory_result['id'],)
            )
            memory_row = cur.fetchone()

        if not memory_row:
            print("❌ Failed to read memory by ID")
            validation_passed = False
        else:
            embedding_model = memory_row['embedding_model']
            expected_model = 'amazon.titan-embed-text-v2:0'

            if embedding_model == expected_model:
                print(f"✓ embedding_model matches: {expected_model}")
            else:
                print(f"❌ embedding_model mismatch: {embedding_model}")
                validation_passed = False

    except Exception as e:
        print(f"❌ Embedding model validation failed: {type(e).__name__}")
        validation_passed = False

    # Close connection
    if db_client:
        db_client.close()
        print()
        print("Step 9: Cleanup")
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
        print("Agent memory successfully persisted and semantically retrieved")
    else:
        print("❌ Some validations FAILED")
        print()
        print("❌ TEST FAILED")


if __name__ == '__main__':
    test_agent_memory()
