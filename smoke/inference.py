import mlflow
import numpy as np
import os
import sys
import argparse

sys.path.append("..")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use Surrogate in inference")
    parser.add_argument("--run-id", type=str)
    args = parser.parse_args()
    print(os.getenv("MLFLOW_TRACKING_URI"))
    logged_model = f"runs:/{args.run_id}/smoke_model"
    # mlflow.pyfunc.get_model_dependencies(logged_model)
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    print(f"{loaded_model=}")
