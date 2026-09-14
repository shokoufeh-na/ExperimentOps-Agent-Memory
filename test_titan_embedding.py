#!/usr/bin/env python3
"""
Test Amazon Titan embedding integration.
Read-only test - does not write to database.
"""

import sys
import os

# Add lambda directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lambda'))

from bedrock_client import BedrockClient


def test_titan_embedding():
    """Test Titan embedding generation."""

    # Test text
    test_text = "Random Forest experiment with learning rate 0.01"

    print(f"Testing Titan embedding for: '{test_text}'")
    print()

    try:
        # Initialize client with experimentops profile
        client = BedrockClient(profile_name='experimentops')
        print("✓ BedrockClient initialized")

        # Generate embedding
        embedding = client.generate_embedding(test_text)
        print("✓ Bedrock request succeeded")
        print()

        # Verify embedding properties
        print("=== Results ===")
        print(f"Model ID: {client.TITAN_EMBED_MODEL}")
        print(f"Embedding type: {type(embedding).__name__}")
        print(f"Embedding dimension: {len(embedding)}")

        # Verify it's a list of numeric values
        if isinstance(embedding, list) and len(embedding) > 0:
            if all(isinstance(x, (int, float)) for x in embedding[:10]):
                print("✓ Embedding is a list of numeric values")
            else:
                print("✗ Embedding contains non-numeric values")
        else:
            print("✗ Embedding is not a valid list")

        # Verify dimension
        if len(embedding) == 1024:
            print("✓ Embedding dimension is exactly 1024")
        else:
            print(f"✗ Expected 1024 dimensions, got {len(embedding)}")

        # Show first 3 values
        print()
        print(f"First 3 values: {embedding[:3]}")

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"❌ Test failed: {error_type}")
        print(f"   {error_msg}")


if __name__ == '__main__':
    test_titan_embedding()
