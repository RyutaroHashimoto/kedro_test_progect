import os
from typing import Any, Dict, List
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

import mlflow


def split_data(data: pd.DataFrame, parameters: Dict) -> List:
    """Splits data into training and test sets.
        Args:
            data: Source data.
            parameters: Parameters defined in parameters.yml.
        Returns:
            A list containing split data.
    """
    target_col = 'Survived'
    X = data.drop(target_col, axis=1).values
    y = data[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )

    return [X_train, X_test, y_train, y_test]


def train_model(X_train: np.ndarray, y_train: np.ndarray, parameters: Dict[str, Any]) -> LinearSVC:

    train_x, valid_x, train_y, valid_y = train_test_split(X_train,
                                                          y_train,
                                                          test_size=parameters['test_size'],
                                                          random_state=parameters['random_state'])

    model = LinearSVC(C=0.1).fit(train_x, train_y)
    return model


def evaluate_model(model: LinearSVC, X_test: np.ndarray, y_test: np.ndarray, parameters: Dict[str, Any]):
    """Calculate the F1 score and log the result.
        Args:
            model: Trained model.
            X_test: Testing data of independent features.
            y_test: Testing data for price.
    """

    prediction = model.predict(X_test)
    prediction = np.where(prediction < 0.5, 0, 1)
    score = round(accuracy_score(y_test, prediction), 3)

    logger = logging.getLogger(__name__)
    logger.info("accuracy score : %.3f.", score)

    file_path = os.path.abspath(__file__).split('/')
    model_name = file_path[-2]
    runName = datetime.now().strftime('%Y%m%d%H%M%S')
    mlflow.start_run(run_name=runName, nested=True)
    mlflow.log_metric("accuracy_score", score)
    mlflow.log_param("test_size",
                     parameters['test_size'])

    mlflow.log_param("random_state",
                     parameters['random_state'])

    mlflow.log_param("model_name",
                     model_name)

    mlflow.end_run()