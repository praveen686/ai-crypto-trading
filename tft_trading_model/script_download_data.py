from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import gc
import glob
import os
import shutil
import sys

from expt_settings.configs import ExperimentConfig
import numpy as np
import pandas as pd
import pyunpack
import wget


# General functions for data downloading & aggregation.
def download_from_url(url, output_path):
    print('Pulling data from {} to {}'.format(url, output_path))
    wget.download(url, output_path)
    print('done')


def recreate_folder(path):
    shutil.rmtree(path)
    os.makedirs(path)


def unzip(zip_path, output_file, data_folder):
    print('Unzipping file: {}'.format(zip_path))
    pyunpack.Archive(zip_path).extractall(data_folder)

    # Checks if unzip was successful
    if not os.path.exists(output_file):
        raise ValueError(
            'Error in unzipping process! {} not found.'.format(output_file))


def download_and_unzip(url, zip_path, csv_path, data_folder):
    download_from_url(url, zip_path)

    unzip(zip_path, csv_path, data_folder)

    print('Done.')


def process_bitcoin(config):

    data_folder = config.data_folder

    # download historical data csv
    csv_path = os.path.join(data_folder, 'bitcoin_his.csv')
    url = 'https://raw.githubusercontent.com/CharlesPeng1998/ai-crypto-trading/develop/datasets/merge/result.csv'
    download_from_url(url, csv_path)

    # download historical data with embeddings
    csv_path = os.path.join(data_folder, 'bitcoin_hisWithEmbs.csv')
    url = 'https://raw.githubusercontent.com/CharlesPeng1998/ai-crypto-trading/develop/datasets/data_embs/resultWithEmbs.csv'
    download_from_url(url, csv_path)

    df = pd.read_csv(csv_path, index_col=0)  # no explicit index
    print('reading bitcoin csv complete. Adding extra inputs')

    # Adds additional date/day fields
    df['Date'] = pd.to_datetime(df['Date'], format="%Y/%m/%d")
    df['days_from_start'] = (df['Date'] -
                             pd.datetime(2014, 1, 1)).apply(lambda x: x.days)
    df['day_of_week'] = df['Date'].apply(lambda x: x.dayofweek)
    df['day_of_month'] = df['Date'].apply(lambda x: x.day)
    df['week_of_year'] = df['Date'].apply(lambda x: x.weekofyear)
    df['month'] = df['Date'].apply(lambda x: x.month)
    df['year'] = df['Date'].apply(lambda x: x.year)

    output_file = config.data_csv_path
    print('Completed formatting, saving to {}'.format(output_file))
    df.to_csv(output_file)

    print('Done.')


# Core routine.
def main(expt_name, force_download, output_folder):
    print('#### Running download script ###')

    expt_config = ExperimentConfig(expt_name, output_folder)

    if os.path.exists(expt_config.data_csv_path) and not force_download:
        print('Data has been processed for {}. Skipping download...'.format(
            expt_name))
        sys.exit(0)
    else:
        print('Resetting data folder...')
        recreate_folder(expt_config.data_folder)

    # Default download functions
    download_functions = {'bitcoin': process_bitcoin}

    if expt_name not in download_functions:
        raise ValueError('Unrecongised experiment! name={}'.format(expt_name))

    download_function = download_functions[expt_name]

    # Run data download
    print('Getting {} data...'.format(expt_name))
    download_function(expt_config)

    print('Download completed.')


if __name__ == '__main__':

    def get_args():
        experiment_names = ExperimentConfig.default_experiments

        parser = argparse.ArgumentParser(description='Data download configs')
        parser.add_argument('expt_name',
                            metavar='e',
                            type=str,
                            nargs='?',
                            choices=experiment_names,
                            help='Experiment Name. Default={}'.format(
                                ','.join(experiment_names)))
        parser.add_argument('output_folder',
                            metavar='f',
                            type=str,
                            nargs='?',
                            default='.',
                            help='Path to folder for data download')
        parser.add_argument('force_download',
                            metavar='r',
                            type=str,
                            nargs='?',
                            choices=['yes', 'no'],
                            default='no',
                            help='Whether to re-run data download')

        args = parser.parse_known_args()[0]

        root_folder = None if args.output_folder == '.' else args.output_folder

        return args.expt_name, args.force_download == 'yes', root_folder

    name, force, folder = get_args()
    main(expt_name=name, force_download=force, output_folder=folder)
