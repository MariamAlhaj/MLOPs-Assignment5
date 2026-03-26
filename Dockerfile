FROM python:3.10-slim

# Accept the MLflow Run ID as a build argument
ARG RUN_ID

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir mlflow scikit-learn

# Copy the trained model download script
COPY train.py .

# Simulate downloading the model from MLflow using the Run ID
RUN echo "Downloading model for Run ID: ${RUN_ID}" && \
    echo "Model artifacts retrieved from MLflow tracking server." && \
    echo "${RUN_ID}" > /app/run_id.txt

# Default command: print the Run ID and start the app
CMD ["sh", "-c", "echo 'Serving model for Run ID:' $(cat /app/run_id.txt) && python -c \"print('Model server running...')\""]
