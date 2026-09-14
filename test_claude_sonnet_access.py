#!/usr/bin/env python3
"""
Diagnostic test for Claude Sonnet 4.5 access via AWS Bedrock.
Tests minimal Converse API call without connecting to database.
"""

import boto3
import json


def test_claude_sonnet_access():
    """Test direct Claude Sonnet access via Bedrock."""

    model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    test_message = "Reply only with hello"

    print("=== Claude Sonnet Access Diagnostic ===")
    print()
    print(f"Model ID: {model_id}")
    print(f"Region: us-west-2")
    print(f"Profile: experimentops")
    print()

    # Initialize Bedrock client
    print("Step 1: Initialize Bedrock client")
    try:
        session = boto3.Session(profile_name='experimentops')
        bedrock_runtime = session.client(
            service_name='bedrock-runtime',
            region_name='us-west-2'
        )
        print("✓ Bedrock runtime client initialized")
    except Exception as e:
        print(f"❌ Client initialization failed: {type(e).__name__}")
        return

    print()

    # Test Claude Sonnet invocation
    print("Step 2: Test Claude Sonnet model invocation")
    print(f"Test message: '{test_message}'")
    print()

    try:
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": test_message
                }
            ]
        }

        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body)
        )

        # Parse response
        response_body = json.loads(response['body'].read())
        response_text = response_body['content'][0]['text']

        print("✓ SUCCESS")
        print()
        print(f"Model ID: {model_id}")
        print(f"Response text: {response_text}")

    except Exception as e:
        print("❌ FAILED")
        print()
        print(f"Exception class: {type(e).__name__}")
        print()

        # Extract safe error details from ClientError
        if hasattr(e, 'response') and isinstance(e.response, dict):
            error_info = e.response.get('Error', {})
            metadata = e.response.get('ResponseMetadata', {})

            error_code = error_info.get('Code', 'N/A')
            error_message = error_info.get('Message', 'N/A')
            http_status = metadata.get('HTTPStatusCode', 'N/A')
            request_id = metadata.get('RequestId', 'N/A')

            print("AWS Error Details:")
            print(f"  Error Code: {error_code}")
            print(f"  HTTP Status: {http_status}")
            print(f"  Request ID: {request_id}")
            print()
            print(f"Error Message:")
            print(f"  {error_message}")
        else:
            print("No additional error details available")

    print()
    print("=== Diagnostic Complete ===")


if __name__ == '__main__':
    test_claude_sonnet_access()
