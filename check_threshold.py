import mlflow
import os
import sys

ACCURACY_THRESHOLD = 0.85

def check_threshold():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    # Read the Run ID from model_info.txt
    model_info_path = "model_info.txt"
    if not os.path.exists(model_info_path):
        print("ERROR: model_info.txt not found.")
        sys.exit(1)

    with open(model_info_path, "r") as f:
        run_id = f.read().strip()

    if not run_id:
        print("ERROR: model_info.txt is empty.")
        sys.exit(1)

    print(f"Checking MLflow Run ID: {run_id}")

    # Fetch the run from MLflow
    client = mlflow.tracking.MlflowClient()
    try:
        run = client.get_run(run_id)
    except Exception as e:
        print(f"ERROR: Could not fetch run from MLflow: {e}")
        sys.exit(1)

    accuracy = run.data.metrics.get("accuracy")
    if accuracy is None:
        print("ERROR: 'accuracy' metric not found in MLflow run.")
        sys.exit(1)

    print(f"Model accuracy: {accuracy:.4f} (threshold: {ACCURACY_THRESHOLD})")

    if accuracy < ACCURACY_THRESHOLD:
        print(f"FAILED: Accuracy {accuracy:.4f} is below threshold {ACCURACY_THRESHOLD}.")
        sys.exit(1)
    else:
        print(f"PASSED: Accuracy {accuracy:.4f} meets the threshold.")
        sys.exit(0)

if __name__ == "__main__":
    check_threshold()
