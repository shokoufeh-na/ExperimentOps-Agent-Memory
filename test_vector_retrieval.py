#!/usr/bin/env python3
"""
Test CockroachDB vector retrieval without inserting data.
Read-only test of the similarity search path.
"""

import sys
import os

# Add lambda directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lambda'))

from bedrock_client import BedrockClient
from db import CockroachDBClient


def test_vector_retrieval():
    """Test vector similarity search on empty experiments table."""

    test_text = "Random Forest experiment with learning rate 0.01"
    print(f"Testing vector retrieval for: '{test_text}'")
    print()

    # Step 1: Generate Titan embedding
    print("=== Step 1: Generate Titan Embedding ===")
    try:
        bedrock = BedrockClient(profile_name='experimentops')
        embedding = bedrock.generate_embedding(test_text)
        print("✓ Titan embedding generation succeeded")
        print(f"  Model: {bedrock.TITAN_EMBED_MODEL}")
        print(f"  Dimension: {len(embedding)}")

        if len(embedding) == 1024:
            print("✓ Embedding dimension is exactly 1024")
        else:
            print(f"✗ Expected 1024 dimensions, got {len(embedding)}")
            return
    except Exception as e:
        print(f"❌ Titan embedding failed: {type(e).__name__}: {e}")
        return

    print()

    # Step 2: Connect to CockroachDB
    print("=== Step 2: CockroachDB Connection ===")

    # Check COCKROACHDB_URL is set (but don't print it)
    if not os.environ.get('COCKROACHDB_URL'):
        print("❌ COCKROACHDB_URL not set")
        return

    db_client = None
    try:
        db_client = CockroachDBClient()
        db_client.connect()
        print("✓ CockroachDB connection succeeded")
    except Exception as e:
        print(f"❌ CockroachDB connection failed: {type(e).__name__}")
        return

    print()

    # Step 3: Perform vector similarity search
    print("=== Step 3: Vector Similarity Search ===")
    try:
        results = db_client.search_similar_experiments(
            query_embedding=embedding,
            limit=5
        )
        print("✓ Vector similarity query succeeded")
        print(f"  Number of results returned: {len(results)}")

        if len(results) == 0:
            print("  ✓ Zero results (expected - experiments table is empty)")
        else:
            print(f"  Found {len(results)} similar experiments")

    except Exception as e:
        print(f"❌ Vector similarity query failed: {type(e).__name__}: {e}")
        return
    finally:
        # Step 4: Close database connection
        if db_client:
            db_client.close()
            print()
            print("=== Step 4: Cleanup ===")
            print("✓ Database connection closed")

    print()
    print("=== Test Summary ===")
    print("✓ Titan embedding generation: SUCCESS")
    print("✓ CockroachDB connection: SUCCESS")
    print("✓ Vector similarity query: SUCCESS")
    print("✓ End-to-end vector retrieval path verified")


if __name__ == '__main__':
    test_vector_retrieval()
