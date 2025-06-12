import pandas as pd
import pytest
from tensorflow.keras.models import Sequential

from train_prof import (read_data,
                   create_model,
                   train_model)


@pytest.fixture
#a característica de fixture é que só é executada qdo realmente for precisar dela
def sample_data():
    """
    A fixture function that returns a sample dataset.

    Returns:
        pandas.DataFrame: A DataFrame containing sample data with three columns: 'feature1',
         'feature2', and 'fetal_health'.
    """
    # Retorna um dataframe
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [6, 7, 8, 9, 10],
        'fetal_health': [1, 1, 2, 3, 2]
    })
    return data


def test_read_data():
    """
    This function tests the `read_data` function. It checks whether the returned data is not
     empty for both features (X) and labels (y).

    Parameters:
    None

    Returns:
    None
    """
    X, y = read_data()
    #Assert, está validando q X não está vazio, ou seja, tem algum dado
    assert not X.empty # Neste momento é q vai ser feito o teste se X não está vazio
    assert not y.empty


def test_create_model():
    """
    Generate the function comment for the given function body in a markdown code block with
    the correct language syntax.
    """
    X, _ = read_data()
    model = create_model(X)

    assert len(model.layers) > 2 #validar se o modelo tem uma camada de entrada e uma de saída
    assert model.trainable # validar se o modelo é treinável
    assert isinstance(model, Sequential) # valida se o modelo é sequencial, ou seja, pode-se fazer mais de uma inserção em um mesmo teste


def test_train_model(sample_data):
    """
    Generate a function comment for the given function body in a markdown code block with
    the correct language syntax.

    Parameters:
        sample_data (pandas.DataFrame): The input data containing features and target
        variable.

    Returns:
        None
    """
    X = sample_data.drop(['fetal_health'], axis=1)
    y = sample_data['fetal_health'] - 1
    model = create_model(X)
    train_model(model, X, y, is_train=False)
    assert model.history.history['loss'][-1] > 0
    assert model.history.history['val_loss'][-1] > 0
