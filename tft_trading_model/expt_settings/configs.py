import os

import data_formatters.bitcoin


class ExperimentConfig(object):
  default_experiments = ['bitcoin']

  def __init__(self, experiment='volatility', root_folder=None):
    if experiment not in self.default_experiments:
      raise ValueError('Unrecognised experiment={}'.format(experiment))

    # Defines all relevant paths
    if root_folder is None:
      root_folder = os.path.join(
          os.path.dirname(os.path.realpath(__file__)), '..', 'outputs')
      print('Using root folder {}'.format(root_folder))

    self.root_folder = root_folder
    self.experiment = experiment
    self.data_folder = os.path.join(root_folder, 'data', experiment)
    self.model_folder = os.path.join(root_folder, 'saved_models', experiment)
    self.results_folder = os.path.join(root_folder, 'results', experiment)

    # Creates folders if they don't exist
    for relevant_directory in [
        self.root_folder, self.data_folder, self.model_folder,
        self.results_folder
    ]:
      if not os.path.exists(relevant_directory):
        os.makedirs(relevant_directory)

  @property
  def data_csv_path(self):
    csv_map = {
        'bitcoin': 'result.csv'
    }

    return os.path.join(self.data_folder, csv_map[self.experiment])

  @property
  def hyperparam_iterations(self):

    return 240 if self.experiment == 'volatility' else 60

  def make_data_formatter(self):
    data_formatter_class = {
        'bitcoin': data_formatters.bitcoin.BitcoinFormatter,
    }

    return data_formatter_class[self.experiment]()
