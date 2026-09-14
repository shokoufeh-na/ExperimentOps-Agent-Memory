"""
AWS Lambda handler for ExperimentOps Agent.
Processes experiment queries and generates recommendations.
"""

import json
import os
import traceback
from typing import Dict, Any, Optional

from bedrock_client import BedrockClient
from db import CockroachDBClient


# Valid experiment statuses
VALID_STATUSES = {'running', 'completed', 'failed'}


def get_aws_profile() -> Optional[str]:
    """
    Get AWS profile for local development.
    Returns None when running in Lambda (uses IAM role).

    Set LOCAL_AWS_PROFILE=experimentops for local development.
    Leave unset in Lambda to use IAM execution role.
    """
    return os.environ.get('LOCAL_AWS_PROFILE')


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for ExperimentOps Agent.

    Expected event structure:
    {
        "action": "query" | "log_experiment",
        "query": "User's experiment query (for action=query)",
        "experiment": {
            "name": "Experiment name",
            "model_name": "Model name",
            "parameters": {...},
            "metrics": {...},
            "status": "running|completed|failed",
            "notes": "...",
            "error_message": "..." (if failed)
        } (for action=log_experiment)
    }

    Returns:
    {
        "statusCode": 200,
        "body": JSON string with recommendation or experiment_id
    }
    """
    try:
        # Initialize Bedrock client
        aws_profile = get_aws_profile()
        bedrock_client = BedrockClient(profile_name=aws_profile)

        action = event.get('action', 'query')

        if action == 'query':
            return handle_query(event, bedrock_client)
        elif action == 'log_experiment':
            return handle_log_experiment(event, bedrock_client)
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': f'Unknown action: {action}',
                    'valid_actions': ['query', 'log_experiment']
                })
            }

    except Exception as e:
        print(f"Error in handler: {str(e)}")
        print(traceback.format_exc())
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Internal server error'})
        }


def handle_query(event: Dict[str, Any], bedrock_client: BedrockClient) -> Dict[str, Any]:
    """
    Handle user query for experiment recommendations.

    Process:
    1. Generate embedding for query
    2. Search for similar experiments
    3. Search for relevant memories
    4. Generate recommendation using Claude
    5. Save recommendation to database
    """
    query = event.get('query')
    if not query:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'query field is required'})
        }

    try:
        # Generate embedding for query
        print(f"Generating embedding for query: {query}")
        query_embedding = bedrock_client.generate_embedding(query)

    except Exception as e:
        print(f"Embedding generation failed: {str(e)}")
        print(traceback.format_exc())
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Failed to generate query embedding'})
        }

    try:
        # Search database for similar experiments and memories
        with CockroachDBClient() as db:
            print("Searching for similar experiments...")
            similar_experiments = db.search_similar_experiments(
                query_embedding,
                limit=5,
                status_filter='completed'  # Only look at successful experiments
            )

            print("Searching for relevant memories...")
            relevant_memories = db.search_similar_memories(
                query_embedding,
                limit=5
            )

            # Generate recommendation using Claude
            print("Generating recommendation with Claude Sonnet 4.5...")
            recommendation_result = bedrock_client.generate_recommendation(
                query,
                similar_experiments,
                relevant_memories
            )

            # Save recommendation to database
            print("Saving recommendation to database...")
            recommendation_id = db.insert_recommendation(
                experiment_id=None,  # Not tied to specific experiment
                recommendation_text=recommendation_result['recommendation'],
                confidence=recommendation_result['confidence'],
                reasoning_model=bedrock_client.CLAUDE_SONNET_MODEL,
                applied=False
            )

            print(f"Recommendation saved with ID: {recommendation_id}")

    except Exception as e:
        print(f"Query processing failed: {str(e)}")
        print(traceback.format_exc())
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Failed to process query'})
        }

    # Return response with structured recommendation
    return {
        'statusCode': 200,
        'body': json.dumps({
            'recommendation': recommendation_result['recommendation'],
            'confidence': recommendation_result['confidence'],
            'suggested_approach': recommendation_result['suggested_approach'],
            'pitfalls': recommendation_result['pitfalls'],
            'hyperparameters': recommendation_result['hyperparameters'],
            'expected_outcomes': recommendation_result['expected_outcomes'],
            'recommendation_id': recommendation_id,
            'similar_experiments_count': len(similar_experiments),
            'relevant_memories_count': len(relevant_memories),
            'context': {
                'similar_experiments': [
                    {
                        'id': exp['id'],
                        'name': exp['experiment_name'],
                        'similarity': exp['similarity']
                    }
                    for exp in similar_experiments
                ],
                'relevant_memories': [
                    {
                        'id': mem['id'],
                        'type': mem['memory_type'],
                        'similarity': mem['similarity']
                    }
                    for mem in relevant_memories
                ]
            }
        })
    }


def handle_log_experiment(event: Dict[str, Any], bedrock_client: BedrockClient) -> Dict[str, Any]:
    """
    Handle logging a new experiment.

    Process:
    1. Extract and validate experiment data
    2. Generate embedding
    3. Store in database
    """
    experiment = event.get('experiment')
    if not experiment:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'experiment field is required'})
        }

    # Validate required fields
    required_fields = ['name', 'status']
    for field in required_fields:
        if field not in experiment:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': f'experiment.{field} is required'})
            }

    # Validate status
    status = experiment['status']
    if status not in VALID_STATUSES:
        return {
            'statusCode': 400,
            'body': json.dumps({
                'error': f'Invalid experiment status: {status}',
                'valid_statuses': list(VALID_STATUSES)
            })
        }

    try:
        # Build text representation for embedding
        embed_text = f"""
        Experiment: {experiment['name']}
        Model: {experiment.get('model_name', 'N/A')}
        Status: {experiment['status']}
        Parameters: {json.dumps(experiment.get('parameters', {}))}
        Metrics: {json.dumps(experiment.get('metrics', {}))}
        Notes: {experiment.get('notes', '')}
        """

        # Generate embedding
        print(f"Generating embedding for experiment: {experiment['name']}")
        embedding = bedrock_client.generate_embedding(embed_text)

    except Exception as e:
        print(f"Embedding generation failed: {str(e)}")
        print(traceback.format_exc())
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Failed to generate experiment embedding'})
        }

    try:
        # Store in database
        with CockroachDBClient() as db:
            print("Inserting experiment into database...")
            experiment_id = db.insert_experiment(
                experiment_name=experiment['name'],  # Map 'name' to 'experiment_name'
                model_name=experiment.get('model_name'),
                parameters=experiment.get('parameters'),
                metrics=experiment.get('metrics'),
                status=experiment['status'],
                notes=experiment.get('notes'),
                error_message=experiment.get('error_message'),
                embedding=embedding
            )

            print(f"Experiment saved with ID: {experiment_id}")

    except Exception as e:
        print(f"Database insertion failed: {str(e)}")
        print(traceback.format_exc())
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Failed to save experiment'})
        }

    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Experiment logged successfully',
            'experiment_id': experiment_id
        })
    }


# For local testing
if __name__ == "__main__":
    # Test query
    test_event = {
        "action": "query",
        "query": "I want to train a BERT model for sentiment analysis. What hyperparameters should I use?"
    }

    result = handler(test_event, None)
    print(json.dumps(json.loads(result['body']), indent=2))
