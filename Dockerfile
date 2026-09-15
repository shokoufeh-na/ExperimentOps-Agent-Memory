# AWS Lambda Python 3.12 runtime
FROM public.ecr.aws/lambda/python:3.12

# Copy requirements and install dependencies
COPY lambda/requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r requirements.txt

# Copy CockroachDB certificate
RUN mkdir -p /opt/cockroach
COPY lambda/certs/root.crt /opt/cockroach/root.crt

# Copy Lambda function code
COPY lambda/handler.py ${LAMBDA_TASK_ROOT}/
COPY lambda/bedrock_client.py ${LAMBDA_TASK_ROOT}/
COPY lambda/db.py ${LAMBDA_TASK_ROOT}/

# Set the Lambda handler
CMD ["handler.handler"]
