from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import shutil
import libs.utils as utils
import numpy as np
import pandas as pd

Deque = collections.deque


class HyperparamOptManager:

  def __init__(self,
               param_ranges,
               fixed_params,
               model_folder,
               override_w_fixed_params=True):
    self.param_ranges = param_ranges

    self._max_tries = 1000
    self.results = pd.DataFrame()
    self.fixed_params = fixed_params
    self.saved_params = pd.DataFrame()

    self.best_score = np.Inf
    self.optimal_name = ""

    # Setup
    # Create folder for saving if its not there
    self.hyperparam_folder = model_folder
    utils.create_folder_if_not_exist(self.hyperparam_folder)

    self._override_w_fixed_params = override_w_fixed_params

  def load_results(self):
    print("Loading results from", self.hyperparam_folder)

    results_file = os.path.join(self.hyperparam_folder, "results.csv")
    params_file = os.path.join(self.hyperparam_folder, "params.csv")

    if os.path.exists(results_file) and os.path.exists(params_file):

      self.results = pd.read_csv(results_file, index_col=0)
      self.saved_params = pd.read_csv(params_file, index_col=0)

      if not self.results.empty:
        self.results.at["loss"] = self.results.loc["loss"].apply(float)
        self.best_score = self.results.loc["loss"].min()

        is_optimal = self.results.loc["loss"] == self.best_score
        self.optimal_name = self.results.T[is_optimal].index[0]

        return True

    return False

  def _get_params_from_name(self, name):
    params = self.saved_params

    selected_params = dict(params[name])

    if self._override_w_fixed_params:
      for k in self.fixed_params:
        selected_params[k] = self.fixed_params[k]

    return selected_params

  def get_best_params(self):
    optimal_name = self.optimal_name

    return self._get_params_from_name(optimal_name)

  def clear(self):
    shutil.rmtree(self.hyperparam_folder)
    os.makedirs(self.hyperparam_folder)
    self.results = pd.DataFrame()
    self.saved_params = pd.DataFrame()

  def _check_params(self, params):
    valid_fields = list(self.param_ranges.keys()) + list(
        self.fixed_params.keys())
    invalid_fields = [k for k in params if k not in valid_fields]
    missing_fields = [k for k in valid_fields if k not in params]

    if invalid_fields:
      raise ValueError("Invalid Fields Found {} - Valid ones are {}".format(
          invalid_fields, valid_fields))
    if missing_fields:
      raise ValueError("Missing Fields Found {} - Valid ones are {}".format(
          missing_fields, valid_fields))

  def _get_name(self, params):
    self._check_params(params)

    fields = list(params.keys())
    fields.sort()

    return "_".join([str(params[k]) for k in fields])

  def get_next_parameters(self, ranges_to_skip=None):
    if ranges_to_skip is None:
      ranges_to_skip = set(self.results.index)

    if not isinstance(self.param_ranges, dict):
      raise ValueError("Only works for random search!")

    param_range_keys = list(self.param_ranges.keys())
    param_range_keys.sort()

    def _get_next():
      parameters = {
          k: np.random.choice(self.param_ranges[k]) for k in param_range_keys
      }

      # Adds fixed params
      for k in self.fixed_params:
        parameters[k] = self.fixed_params[k]

      return parameters

    for _ in range(self._max_tries):

      parameters = _get_next()
      name = self._get_name(parameters)

      if name not in ranges_to_skip:
        return parameters

    raise ValueError("Exceeded max number of hyperparameter searches!!")

  def update_score(self, parameters, loss, model, info=""):
    if np.isnan(loss):
      loss = np.Inf

    if not os.path.isdir(self.hyperparam_folder):
      os.makedirs(self.hyperparam_folder)

    name = self._get_name(parameters)

    is_optimal = self.results.empty or loss < self.best_score

    # save the first model
    if is_optimal:
      # Try saving first, before updating info
      if model is not None:
        print("Optimal model found, updating")
        model.save(self.hyperparam_folder)
      self.best_score = loss
      self.optimal_name = name

    self.results[name] = pd.Series({"loss": loss, "info": info})
    self.saved_params[name] = pd.Series(parameters)

    self.results.to_csv(os.path.join(self.hyperparam_folder, "results.csv"))
    self.saved_params.to_csv(os.path.join(self.hyperparam_folder, "params.csv"))

    return is_optimal