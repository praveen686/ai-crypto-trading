import os
import numpy as np

import tensorflow.compat.v1 as tf
import libs.hyperparam_opt
import libs.tft_model
import libs.utils as utils
import expt_settings.configs
import pandas as pd

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
        self.quantiles = [0.1, 0.5, 0.9]

        if fixed_param:
            model_folder = os.path.join(self.config.model_folder, "fixed")
            fixed_params = self.formatter.get_experiment_params()
            params = self.formatter.get_default_model_params()
            params["model_folder"] = model_folder
            self.opt_manager = HyperparamOptManager(
                {k: [params[k]]
                 for k in params}, fixed_params, model_folder, False)
        else:
            model_folder = os.path.join(self.config.model_folder, "main")
            fixed_params = self.formatter.get_experiment_params()
            param_ranges = ModelClass.get_hyperparm_choices()
            fixed_params["model_folder"] = model_folder
            self.opt_manager = HyperparamOptManager(param_ranges, fixed_params,
                                                    model_folder, False)

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
            success = self.opt_manager.load_results()
            best_params = self.opt_manager.get_best_params()
            model = ModelClass(best_params, use_cudnn=self.use_gpu)

            model.load(self.opt_manager.hyperparam_folder)

            combined = model.model.predict(inputs,
                                           workers=16,
                                           use_multiprocessing=True,
                                           batch_size=inputs.shape[0])

            process_map = {
                'p{}'.format(int(q * 100)): combined[Ellipsis, i:(i + 1)]
                for i, q in enumerate(self.quantiles)
            }

        return process_map

    def execute_stratedy(self, pre):
        data_folder = self.config.data_folder
        csv_path = os.path.join(data_folder, 'bitcoin_his.csv')
        df = pd.read_csv(csv_path, index_col=0)  # no explicit index
        #print(df)
        principle = [10000 for _ in range(7)]
        position = [0 for _ in range(7)]
        cost = [0 for _ in range(7)]
        n = len(pre)
        for index in range(7, n):
            curr_price = df.at[index-1, 'market_price']
            pred_price = pre['p90'][index-7][0][0]
            # execute buy
            if position[-1] == 0:
                if pred_price > curr_price:
                    num = principle[-1]/curr_price
                    position.append(num)
                    principle.append(0)
                    cost.append(curr_price)
                else:
                    position.append(0)
                    principle.append(principle[-1])
                    cost.append(0)
            else: # execute sell
                if pred_price < curr_price or round(curr_price/cost[-1]-1, 4) > 0.05 or round(curr_price/cost[-1]-1, 4) < -0.03:
                    principle.append(position[-1]*curr_price)
                    position.append(0)
                    cost.append(0)
                else:
                    position.append(position[-1])
                    principle.append(0)
                    cost.append(cost[-1])
        df['principle'] = principle
        df['position'] = position
        df['cost'] = cost
        print(df)
        
    def get_input_data(self):
        data_folder = self.config.data_folder
        csv_path = os.path.join(data_folder, 'bitcoin_hisWithEmbs.csv')
        df = pd.read_csv(csv_path, index_col=0)  # no explicit index
        return self.formatter.get_all_data(df)


if __name__ == '__main__':
    expt_name = "bitcoin"
    output_folder = "../../tft_script"
    use_gpu = True

    tft_strategy = TFTStrategy(expt_name, output_folder, use_gpu)
    #inputs = np.random.randn(1, 10, 777)
    inputs = tft_strategy.get_input_data()
    predict_result = tft_strategy.predict_batch(inputs)
    print(tft_strategy.formatter.format_predictions(predict_result))
