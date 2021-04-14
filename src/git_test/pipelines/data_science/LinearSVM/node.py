import pandas as pd
from sklearn import preprocessing


def _label_encoding(df: pd.DataFrame) -> (pd.DataFrame, dict):

    df_le = df.copy()
    list_columns_object = df_le.columns[df_le.dtypes == 'object']

    dict_encoders = {}
    for column in list_columns_object:
        le = preprocessing.LabelEncoder()
        mask_nan = df_le[column].isnull()
        df_le[column] = le.fit_transform(df_le[column].fillna('NaN'))

        df_le.loc[mask_nan, column] *= -1
        dict_encoders[column] = le

    return df_le, dict_encoders


def _drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_prep = df.copy()
    drop_cols = ['Name', 'Ticket', 'PassengerId']
    df_prep = df_prep.drop(drop_cols, axis=1)
    return df_prep


def _process_Age_column(series: pd.Series) -> pd.Series:
    series_prep = series.copy()
    series_prep = series_prep.fillna(series_prep.mean())
    return series_prep


def _process_Embarked_column(series: pd.Series) -> pd.Series:
    series_prep = series.copy()
    series_prep = series_prep.fillna(series_prep.mode()[0])
    return series_prep


def _process_Pclass_column(series: pd.Series) -> pd.Series:
    series_prep = series.copy()
    series_prep = series_prep.astype(str)
    return series_prep


def _process_Cabin_column(series: pd.Series) -> pd.Series:
    series_prep = series.copy()
    series_prep = series_prep.str[0]
    return series_prep


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df_prep = df.copy()

    df_prep = _drop_columns(df_prep)
    df_prep['Age'] = _process_Age_column(df_prep['Age'])
    df_prep['Embarked'] = _process_Embarked_column(df_prep['Embarked'])
    df_prep['Pclass'] = _process_Pclass_column(df_prep['Pclass'])
    df_prep['Cabin'] = _process_Cabin_column(df_prep['Cabin'])

    df_prep, _ = _label_encoding(df_prep)

    return df_prep