#!/usr/bin/env python3
"""
Enhanced diagnostic that reproduces generate_recommendation() exactly.
Tests the complete recommendation pipeline with detailed error reporting.
"""

import sys
import os
import json
import boto3

# Add lambda directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lambda'))

from bedrock_client import BedrockClient
from db import CockroachDBClient


def format_experiments(experiments):
    """Format experiments for context (mimics _format_experiments)."""
    if not experiments:
        return "No similar experiments found."

    formatted = []
    for i, exp in enumerate(experiments, 1):
        formatted.append(
            f"{i}. {exp['experiment_name']}\n"
            f"   Model: {exp.get('model_name', 'N/A')}\n"
            f"   Status: {exp['status']}\n"
            f"   Metrics: {json.dumps(exp.get('metrics', {}), indent=2)}\n"
            f"   Notes: {exp.get('notes', 'N/A')}"
        )

    return "\n\n".join(formatted)


def format_memories(memories):
    """Format agent memories for context (mimics _format_memories)."""
    if not memories:
        return "No relevant memories found."

    formatted = []
    for i, mem in enumerate(memories, 1):
        formatted.append(
            f"{i}. [{mem['memory_type']}] {mem['content']}\n"
            f"   Context: {json.dumps(mem.get('context', {}))}"
        )

    return "\n\n".join(formatted)


def build_prompt(query, experiments_context, memories_context):
    """Build prompt for Claude (mimics _build_prompt)."""
    return f"""You are an expert ML experiment advisor for ExperimentOps Agent.

User Query:
{query}

Similar Past Experiments:
{experiments_context}

Relevant Agent Memories:
{memories_context}

Based on the above context, provide a detailed recommendation for the user's experiment.

You MUST respond with valid JSON in this exact structure:
{{
  "recommendation": "Overall recommendation summary as a single paragraph",
  "confidence": 0.0,
  "suggested_approach": "Detailed approach based on past experiments",
  "pitfalls": ["Pitfall 1", "Pitfall 2", "..."],
  "hyperparameters": {{"param1": "value", "param2": "value"}},
  "expected_outcomes": "What the user should expect"
}}

Requirements:
- confidence must be a number between 0.0 and 1.0
- pitfalls must be an array of strings
- hyperparameters must be an object/dict
- All other fields must be strings
- Return ONLY the JSON, no other text"""


def test_recommendation_diagnostic():
    """Enhanced diagnostic reproducing exact generate_recommendation() flow."""

    user_query = "I need to detect suspicious network traffic. Which previous experiment should I build on, and what should I try next?"

    print("=== Recommendation Diagnostic (Exact Flow Reproduction) ===")
    print()
    print(f"User Query: '{user_query}'")
    print()

    # Step 1: Initialize BedrockClient
    print("Step 1: Initialize BedrockClient")

    try:
        bedrock_client = BedrockClient(profile_name='experimentops')
        print("✓ BedrockClient initialized")
    except Exception as e:
        print(f"❌ BedrockClient initialization failed: {type(e).__name__}")
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
        print("✓ CockroachDB connected")
    except Exception as e:
        print(f"❌ Database connection failed: {type(e).__name__}")
        return

    print()

    # Step 3: Generate query embedding and retrieve experiments
    print("Step 3: Retrieve top 3 similar experiments")

    try:
        query_embedding = bedrock_client.generate_embedding(user_query)
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

        print()
        print("Retrieved experiments:")
        for i, exp in enumerate(similar_experiments, 1):
            print(f"  {i}. {exp['experiment_name']}")
            print(f"     Model: {exp.get('model_name', 'N/A')}")
            print(f"     Similarity: {exp.get('similarity', 0):.4f}")

        print()

        # HARD REQUIREMENT: Network Anomaly Detection must be present
        found_target = any(
            exp['experiment_name'] == "Network Anomaly Detection"
            for exp in similar_experiments
        )

        if not found_target:
            print("❌ TEST FAILED: 'Network Anomaly Detection' not found in top 3 results")
            if db_client:
                db_client.close()
            return

        print("✓ 'Network Anomaly Detection' found in top 3")

    except Exception as e:
        print(f"❌ Experiment retrieval failed: {type(e).__name__}")
        if db_client:
            db_client.close()
        return

    print()

    # Step 4: Format context and build prompt
    print("Step 4: Format context and build prompt")

    try:
        experiments_context = format_experiments(similar_experiments)
        memories_context = format_memories([])  # Empty list for now
        prompt = build_prompt(user_query, experiments_context, memories_context)

        print("✓ Prompt built")
        print(f"  Prompt length: {len(prompt)} characters")

    except Exception as e:
        print(f"❌ Prompt building failed: {type(e).__name__}")
        if db_client:
            db_client.close()
        return

    print()

    # Step 5: Invoke Claude Sonnet via invoke_model (exact reproduction)
    print("Step 5: Invoke Claude Sonnet via bedrock_runtime.invoke_model()")
    print(f"  Model ID: us.anthropic.claude-sonnet-4-5-20250929-v1:0")

    try:
        # Build exact request body
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "temperature": 0.7,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        # Invoke model (exact reproduction)
        response = bedrock_client.bedrock_runtime.invoke_model(
            modelId="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body)
        )

        print("✓ Bedrock invoke_model() call succeeded")

    except Exception as e:
        print(f"❌ Bedrock invoke_model() failed: {type(e).__name__}")

        if hasattr(e, 'response') and isinstance(e.response, dict):
            error_info = e.response.get('Error', {})
            metadata = e.response.get('ResponseMetadata', {})

            print()
            print("AWS Error Details:")
            print(f"  Error Code: {error_info.get('Code', 'N/A')}")
            print(f"  HTTP Status: {metadata.get('HTTPStatusCode', 'N/A')}")
            print(f"  Request ID: {metadata.get('RequestId', 'N/A')}")
            print(f"  Message: {error_info.get('Message', 'N/A')}")

        if db_client:
            db_client.close()
        return

    print()

    # Step 6: Parse outer Bedrock response
    print("Step 6: Parse outer Bedrock response")

    try:
        response_body = json.loads(response['body'].read())
        print("✓ Outer response JSON parsed")

    except Exception as e:
        print(f"❌ Outer response parsing failed: {type(e).__name__}")
        if db_client:
            db_client.close()
        return

    print()

    # Step 7: Extract recommendation text
    print("Step 7: Extract text from response['content'][0]['text']")

    try:
        recommendation_text = response_body['content'][0]['text']
        print("✓ Text extracted successfully")

    except Exception as e:
        print(f"❌ Text extraction failed: {type(e).__name__}")
        print(f"  Response body keys: {list(response_body.keys())}")
        if db_client:
            db_client.close()
        return

    print()

    # Step 8: Display raw model response
    print("Step 8: Raw Claude Sonnet Response")
    print("=" * 70)
    print(recommendation_text)
    print("=" * 70)
    print()

    # Step 9: Parse recommendation JSON
    print("Step 9: Parse recommendation JSON (json.loads)")

    try:
        recommendation_data = json.loads(recommendation_text)
        print("✓ Recommendation JSON parsed successfully")

    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {type(e).__name__}")
        print(f"  Error position: {e.pos}")
        print(f"  Error line: {e.lineno}, column: {e.colno}")
        if db_client:
            db_client.close()
        return

    print()

    # Step 10: Validate required fields
    print("Step 10: Validate required fields")

    required_fields = [
        'recommendation',
        'confidence',
        'suggested_approach',
        'pitfalls',
        'hyperparameters',
        'expected_outcomes'
    ]

    missing_fields = [f for f in required_fields if f not in recommendation_data]

    if missing_fields:
        print(f"❌ Missing required fields: {missing_fields}")
        print(f"  Present fields: {list(recommendation_data.keys())}")
    else:
        print("✓ All required fields present")

    print()

    # Step 11: Validate confidence
    print("Step 11: Validate confidence")

    confidence = recommendation_data.get('confidence', None)

    print(f"  Confidence value: {confidence}")
    print(f"  Confidence type: {type(confidence).__name__}")

    confidence_valid = isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0

    if confidence_valid:
        print("✓ Confidence is valid (numeric, 0.0-1.0)")
    else:
        print(f"❌ Confidence is invalid")

    print()

    # Step 12: Display field types
    print("Step 12: Field type validation")

    for field in required_fields:
        if field in recommendation_data:
            field_type = type(recommendation_data[field]).__name__
            print(f"  {field}: {field_type}")

    print()

    # Close connection
    if db_client:
        db_client.close()
        print("Step 13: Cleanup")
        print("✓ Database connection closed")
        print()

    # Final summary
    print("=" * 70)
    print("=== Diagnostic Summary ===")
    print()

    all_passed = (
        len(missing_fields) == 0 and
        confidence_valid
    )

    if all_passed:
        print("✓ All validation steps PASSED")
        print()
        print("✓ RECOMMENDATION PIPELINE IS WORKING CORRECTLY")
        print()
        print("The generate_recommendation() method should work as expected.")
    else:
        print("❌ Some validation steps FAILED")
        print()
        if missing_fields:
            print(f"  - Missing fields: {missing_fields}")
        if not confidence_valid:
            print(f"  - Invalid confidence: {confidence}")
        print()
        print("❌ RECOMMENDATION PIPELINE HAS ISSUES")


if __name__ == '__main__':
    test_recommendation_diagnostic()
