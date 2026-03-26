import mlflow
import os
import sys
import dagshub

ACCURACY_THRESHOLD = 0.85

def check_threshold():
    username = os.environ.get("MLFLOW_TRACKING_USERNAME")
    password = os.environ.get("MLFLOW_TRACKING_PASSWORD")

    dagshub.auth.add_app_token(password)
    dagshub.init(
        repo_owner=username,
        repo_name="MLOPs-Assignment5",
        mlflow=True
    )

    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()

    print(f"Checking MLflow Run ID: {run_id}")

    client = mlflow.tracking.MlflowClient()
    try:
        run = client.get_run(run_id)
    except Exception as e:
        print(f"ERROR: Could not fetch run from MLflow: {e}")
        sys.exit(1)

    accuracy = run.data.metrics.get("accuracy")
    print(f"Model accuracy: {accuracy:.4f} (threshold: {ACCURACY_THRESHOLD})")

    if accuracy < ACCURACY_THRESHOLD:
        print(f"FAILED: Accuracy {accuracy:.4f} is below threshold {ACCURACY_THRESHOLD}.")
        sys.exit(1)
    else:
        print(f"PASSED: Accuracy {accuracy:.4f} meets the threshold.")
        sys.exit(0)

if __name__ == "__main__":
    check_threshold()