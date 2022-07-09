import os
import numpy as np

import tensorflow.compat.v1 as tf
import libs.tft_model
import libs.utils as utils
import expt_settings.configs

ExperimentConfig = expt_settings.configs.ExperimentConfig
HyperparamOptManager = libs.hyperparam_opt.HyperparamOptManager
ModelClass = libs.tft_model.TemporalFusionTransformer


class TFTStrategy:

    def __init__(self, expt_name, output_folder, use_gpu, fixed_param=True):
        self.config = ExperimentConfig(expt_name, output_folder)
        self.formatter = self.config.make_data_formatter()
        self.use_gpu = use_gpu
        self.opt_manager = None
        self.tf_config = None

        if fixed_param:
            model_folder = os.path.join(self.config.model_folder, "fixed")
            fixed_params = self.formatter.get_experiment_params()
            params = self.formatter.get_default_model_params()
            params["model_folder"] = model_folder
            self.opt_manager = HyperparamOptManager(
                {k: [params[k]]
                 for k in params}, fixed_params, model_folder)
        else:
            model_folder = os.path.join(self.config.model_folder, "main")
            fixed_params = self.formatter.get_experiment_params()
            param_ranges = ModelClass.get_hyperparm_choices()
            fixed_params["model_folder"] = model_folder
            self.opt_manager = HyperparamOptManager(param_ranges, fixed_params,
                                                    model_folder)

        if use_gpu:
            self.tf_config = utils.get_default_tensorflow_config(
                tf_device="gpu", gpu_id=0)
        else:
            self.tf_config = utils.get_default_tensorflow_config(
                tf_device="cpu")

    def predict_batch(self, inputs):
        """
        The inputs should have shape of (batch_size, time_steps, input_size)
        """
        print("****** Running prediction ******")
        print("The shape of inputs: {}".format(inputs.shape))

        tf.reset_default_graph()
        with tf.Graph().as_default(), tf.Session(
                config=self.tf_config) as sess:
            tf.keras.backend.set_session(sess)
            best_params = self.opt_manager.get_best_params()
            model = ModelClass(best_params, use_cudnn=self.use_gpu)

            model.load(self.opt_manager.hyperparam_folder)

            combined = model.model.predict(inputs,
                                           workers=16,
                                           use_multiprocessing=True,
                                           batch_size=inputs.shape[0])

            process_map = {
                'p{}'.format(int(q * 100)):
                combined[Ellipsis,
                         i * self.output_size:(i + 1) * self.output_size]
                for i, q in enumerate(self.quantiles)
            }

            print("Predict result: {}".format(process_map))


if __name__ == '__main__':
    expt_name = "bitcoin"
    output_folder = "../../tft_script"
    use_gpu = True

    tft_strategy = TFTStrategy(expt_name, output_folder, use_gpu)
    inputs = np.random.randn(1, 10, 4)
