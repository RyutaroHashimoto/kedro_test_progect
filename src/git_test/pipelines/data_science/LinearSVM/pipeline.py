from kedro.pipeline import node, Pipeline
from .node import split_data, train_model, evaluate_model


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["train_prep", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
            ),
            node(
                 func=train_model,
                 inputs=["X_train", "y_train", "parameters"],
                 outputs="model"),
            node(
                func=evaluate_model,
                inputs=["model", "X_test", "y_test", "parameters"],
                outputs=None,
            ),
        ],
    )