import os
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


# Generic.
def get_single_col_by_input_type(input_type, column_definition):
    l = [tup[0] for tup in column_definition if tup[2] == input_type]

    if len(l) != 1:
        raise ValueError('Invalid number of columns for {}'.format(input_type))

    return l[0]


def extract_cols_from_data_type(data_type, column_definition,
                                excluded_input_types):
    return [
        tup[0] for tup in column_definition
        if tup[1] == data_type and tup[2] not in excluded_input_types
    ]


# Loss functions.
def tensorflow_quantile_loss(y, y_pred, quantile):
    # Checks quantile
    if quantile < 0 or quantile > 1:
        raise ValueError(
            'Illegal quantile value={}! Values should be between 0 and 1.'.
            format(quantile))

    prediction_underflow = y - y_pred
    q_loss = quantile * tf.maximum(prediction_underflow, 0.) + (
        1. - quantile) * tf.maximum(-prediction_underflow, 0.)

    return tf.reduce_sum(q_loss, axis=-1)


def numpy_normalised_quantile_loss(y, y_pred, quantile):
    prediction_underflow = y - y_pred
    weighted_errors = quantile * np.maximum(prediction_underflow, 0.) \
        + (1. - quantile) * np.maximum(-prediction_underflow, 0.)

    quantile_loss = weighted_errors.mean()
    normaliser = y.abs().mean()

    return 2 * quantile_loss / normaliser


# OS related functions.
def create_folder_if_not_exist(directory):
    # Also creates directories recursively
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


# Tensorflow related functions.
def get_default_tensorflow_config(tf_device='gpu', gpu_id=0):
    if tf_device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # for training on cpu
        tf_config = tf.ConfigProto(log_device_placement=False,
                                   device_count={'GPU': 0})

    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        print('Selecting GPU ID={}'.format(gpu_id))

        tf_config = tf.ConfigProto(log_device_placement=False)
        tf_config.gpu_options.allow_growth = True

    return tf_config


def save(tf_session, model_folder, cp_name, scope=None):
    # Save model
    if scope is None:
        saver = tf.train.Saver()
    else:
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope=scope)
        saver = tf.train.Saver(var_list=var_list, max_to_keep=100000)

    save_path = saver.save(
        tf_session, os.path.join(model_folder, '{0}.ckpt'.format(cp_name)))
    print('Model saved to: {0}'.format(save_path))


def load(tf_session, model_folder, cp_name, scope=None, verbose=False):
    # Load model proper
    load_path = os.path.join(model_folder, '{0}.ckpt'.format(cp_name))

    print('Loading model from {0}'.format(load_path))

    print_weights_in_checkpoint(model_folder, cp_name)

    initial_vars = set(
        [v.name for v in tf.get_default_graph().as_graph_def().node])

    # Saver
    if scope is None:
        saver = tf.train.Saver()
    else:
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                     scope=scope)
        saver = tf.train.Saver(var_list=var_list, max_to_keep=100000)
    # Load
    saver.restore(tf_session, load_path)
    all_vars = set(
        [v.name for v in tf.get_default_graph().as_graph_def().node])

    if verbose:
        print('Restored {0}'.format(','.join(
            initial_vars.difference(all_vars))))
        print('Existing {0}'.format(','.join(
            all_vars.difference(initial_vars))))
        print('All {0}'.format(','.join(all_vars)))

    print('Done.')


def print_weights_in_checkpoint(model_folder, cp_name):
    load_path = os.path.join(model_folder, '{0}.ckpt'.format(cp_name))

    print_tensors_in_checkpoint_file(file_name=load_path,
                                     tensor_name='',
                                     all_tensors=True,
                                     all_tensor_names=True)
