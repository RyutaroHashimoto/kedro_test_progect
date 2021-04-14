from kedro.pipeline import node, Pipeline
from .node import preprocess

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess,
                inputs="train",
                outputs="train_prep",
                name="preprocess",
            ),
        ],
    )