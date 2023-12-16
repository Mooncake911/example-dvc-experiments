import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Text
import yaml


def featurize(config_path: Text) -> None:
    """Create new features.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    dataset = pd.read_csv(config['data_load']['dataset_csv'])
    target_column = config['featurize']['target_column']

    dataset['sepal_length_to_sepal_width'] = dataset['sepal_length'] / dataset['sepal_width']
    dataset['petal_length_to_petal_width'] = dataset['petal_length'] / dataset['petal_width']

    scaler = StandardScaler()
    featured_dataset = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)
    featured_dataset[target_column] = dataset[target_column]

    features_path = config['featurize']['features_path']
    featured_dataset.to_csv(features_path, index=False)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    featurize(config_path=args.config)
