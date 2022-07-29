import argparse
import datetime as dte
import os

import data_formatters.base
import expt_settings.configs
import libs.hyperparam_opt
import libs.tft_model
import libs.utils as utils
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

ExperimentConfig = expt_settings.configs.ExperimentConfig
HyperparamOptManager = libs.hyperparam_opt.HyperparamOptManager
ModelClass = libs.tft_model.TemporalFusionTransformer


def main(expt_name,
         use_gpu,
         model_folder,
         data_csv_path,
         data_formatter,
         use_testing_mode=False):
  num_repeats = 1

  if not isinstance(data_formatter, data_formatters.base.GenericDataFormatter):
    raise ValueError(
        "Data formatters should inherit from" +
        "AbstractDataFormatter! Type={}".format(type(data_formatter)))

  # Tensorflow setup
  default_keras_session = tf.keras.backend.get_session()

  if use_gpu:
    tf_config = utils.get_default_tensorflow_config(tf_device="gpu", gpu_id=0)

  else:
    tf_config = utils.get_default_tensorflow_config(tf_device="cpu")

  print("*** Training from defined parameters for {} ***".format(expt_name))

  print("Loading & splitting data...")
  raw_data = pd.read_csv(data_csv_path, index_col=0)
  train, valid, test = data_formatter.split_data(raw_data)

  train_samples, valid_samples = data_formatter.get_num_samples_for_calibration(
  )

  # Sets up default params
  fixed_params = data_formatter.get_experiment_params()
  params = data_formatter.get_default_model_params()
  params["model_folder"] = model_folder

  # Parameter overrides for testing only! Small sizes used to speed up script.
  if use_testing_mode:
    fixed_params["num_epochs"] = 1
    params["hidden_layer_size"] = 5
    train_samples, valid_samples = 100, 10

  # Sets up hyperparam manager
  print("*** Loading hyperparm manager ***")
  opt_manager = HyperparamOptManager({k: [params[k]] for k in params},
                                     fixed_params, model_folder)

  # Training -- one iteration only
  print("*** Running calibration ***")
  print("Params Selected:")
  for k in params:
    print("{}: {}".format(k, params[k]))

  best_loss = np.Inf
  for _ in range(num_repeats):

    tf.reset_default_graph()
    with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:

      tf.keras.backend.set_session(sess)

      params = opt_manager.get_next_parameters()
      model = ModelClass(params, use_cudnn=use_gpu)

      if not model.training_data_cached():
        model.cache_batched_data(train, "train", num_samples=train_samples)
        model.cache_batched_data(valid, "valid", num_samples=valid_samples)

      sess.run(tf.global_variables_initializer())
      model.fit()

      val_loss = model.evaluate()

      if val_loss < best_loss:
        opt_manager.update_score(params, val_loss, model)
        best_loss = val_loss

      tf.keras.backend.set_session(default_keras_session)

if __name__ == "__main__":

  def get_args():
    experiment_names = ExperimentConfig.default_experiments

    parser = argparse.ArgumentParser(description="Data download configs")
    parser.add_argument(
        "expt_name",
        metavar="e",
        type=str,
        nargs="?",
        default="volatility",
        choices=experiment_names,
        help="Experiment Name. Default={}".format(",".join(experiment_names)))
    parser.add_argument(
        "output_folder",
        metavar="f",
        type=str,
        nargs="?",
        default=".",
        help="Path to folder for data download")
    parser.add_argument(
        "use_gpu",
        metavar="g",
        type=str,
        nargs="?",
        choices=["yes", "no"],
        default="no",
        help="Whether to use gpu for training.")

    args = parser.parse_known_args()[0]

    root_folder = None if args.output_folder == "." else args.output_folder

    return args.expt_name, root_folder, args.use_gpu == "yes"

  name, output_folder, use_tensorflow_with_gpu = get_args()

  print("Using output folder {}".format(output_folder))

  config = ExperimentConfig(name, output_folder)
  formatter = config.make_data_formatter()

  main(
      expt_name=name,
      use_gpu=use_tensorflow_with_gpu,
      model_folder=os.path.join(config.model_folder, "fixed"),
      data_csv_path=config.data_csv_path,
      data_formatter=formatter,
      use_testing_mode=False)  
