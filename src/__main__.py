import os
import random

import mlflow


def setup_mlflow():
    # load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("buc_factory")


if __name__ == "__main__":
    setup_mlflow()
    with mlflow.start_run(run_name="example_run"):
        lr = 0.01
        acc = random.uniform(0.8, 0.95)

        mlflow.log_param("learning_rate", lr)
        mlflow.log_metric("accuracy", acc)

    print("Run logged.")
