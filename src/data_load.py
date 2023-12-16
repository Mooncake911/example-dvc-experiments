import argparse
from sklearn.datasets import load_iris
from typing import Text
import yaml


def data_load(config_path: Text) -> None:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    data = load_iris(as_frame=True)
    dataset = data.frame
    dataset.rename(
        columns=lambda colname: colname.strip(' (cm)').replace(' ', '_'),
        inplace=True
    )

    dataset.to_csv(config['data_load']['dataset_csv'], index=False)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_load(config_path=args.config)
