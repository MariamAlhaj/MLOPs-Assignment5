import mlflow
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

def train():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    username = os.environ.get("MLFLOW_TRACKING_USERNAME")
    password = os.environ.get("MLFLOW_TRACKING_PASSWORD")

    # Embed credentials directly into the tracking URI
    # https://username:password@dagshub.com/...
    uri_with_creds = tracking_uri.replace("https://", f"https://{username}:{password}@")
    mlflow.set_tracking_uri(uri_with_creds)

    mlflow.set_experiment("assignment5")

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run() as run:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", accuracy)

        with open("model.pkl", "wb") as f:
            pickle.dump(clf, f)
        mlflow.log_artifact("model.pkl")

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Run ID: {run.info.run_id}")

        with open("model_info.txt", "w") as f:
            f.write(run.info.run_id)

        print("model_info.txt written successfully.")

    return accuracy

if __name__ == "__main__":
    train()