if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

import pickle
import mlflow

from mlflow.tracking import MlflowClient


EXPERIMENT_NAME = "linear-regression-models"

mlflow.set_tracking_uri('http://mlflow:5000')
mlflow.set_experiment(EXPERIMENT_NAME)

@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    dv, lr = data

    artifact_name = 'dict_vectorizer.bin'

    with mlflow.start_run():
        with open(artifact_name, 'wb') as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact(artifact_name)
        mlflow.sklearn.log_model(lr, 'model')

    print('Done!')

