mlflow server \
  --backend-store-uri sqlite:///mlflow_data/mlflow.db \
  --default-artifact-root file:///mlflow_data/artifacts \
  --host 127.0.0.1 \
  --port 5000