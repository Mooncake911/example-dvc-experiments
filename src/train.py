import os
import argparse
import pandas as pd
from typing import Text
import yaml

import tensorflow as tf
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from dvclive import Live
from dvclive.keras import DVCLiveCallback


def draw_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('plots/confusion_matrix.png')


def get_model(optimizer, loss, metrics):

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3)
    ])

    optimizer = optimizer if optimizer is not None else 'adam'
    loss = loss if loss is not None else tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = metrics if metrics is not None else ['accuracy', 'mae']
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model


def train(config_path: Text) -> None:
    """Train model.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    # Определяем модель
    m = get_model(optimizer=config['model']['optimizer'],
                  loss=config['model']['loss'],
                  metrics=config['model']['metrics'])

    # Подготавливаем данные
    train_df = pd.read_csv(config['featurize']['features_path'])
    target_column = config['featurize']['target_column']

    x_train, x_test, y_train, y_test = train_test_split(
        train_df, train_df[target_column],
        test_size=config['data_split']['test_size'],
        random_state=config['base']['random_state']
    )

    batch_size = config['train']['batch_size']
    epochs = config['train']['epochs']

    # Вычисляем модель
    with Live() as live:
        m.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test),
            callbacks=[DVCLiveCallback(live=live)],
        )

        test_loss, test_acc, test_mae = m.evaluate(x_test)
        live.log_metric("test_loss", test_loss, plot=False)
        live.log_metric("test_acc", test_acc, plot=False)
        live.log_metric("test_mae", test_acc, plot=False)

        y_prob = m.predict(x_test)
        y_pred = y_prob.argmax(axis=-1)

        live.log_sklearn_plot("confusion_matrix", y_test, y_pred, name="cm.json")

        df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred,
        })
        os.makedirs("plots")
        df.to_csv('plots/confusion.csv', index=False)
        draw_confusion_matrix(y_test, y_pred)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train(config_path=args.config)
