import os
from sre_parse import fix_flags
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
        self.fixed_param = fixed_param 

        if use_gpu:
            self.tf_config = utils.get_default_tensorflow_config(
                tf_device="gpu", gpu_id=0)
        else:
            self.tf_config = utils.get_default_tensorflow_config(
                tf_device="cpu")

    def predict_all(self):
        raw_data = pd.read_csv(self.config.data_csv_path, index_col=0)
        inputs = self.formatter.get_all_data(raw_data)

        tf.reset_default_graph()
        with tf.Graph().as_default(), tf.Session(
                config=self.tf_config) as sess:
            tf.keras.backend.set_session(sess)

            if self.fixed_param:
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

            success = self.opt_manager.load_results()
            best_params = self.opt_manager.get_best_params()
            model = ModelClass(best_params, use_cudnn=self.use_gpu)
            model.load(self.opt_manager.hyperparam_folder)

            
            output_map = model.predict(inputs, return_targets=True)
            p50_forecast = self.formatter.format_predictions(output_map["p50"])
            p90_forecast = self.formatter.format_predictions(output_map["p90"])
            p90_forecast.to_csv("output.csv")

    def execute_strategy(self):
        data_folder = self.config.data_folder
        csv_path = os.path.join(data_folder, 'bitcoin_his.csv')
        data = pd.read_csv(csv_path, index_col=False)  # no explicit index
        data['Date'] = pd.to_datetime(data['Date'])
        data['Date'] = data['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        pre = pd.read_csv('output.csv', index_col=0)
        #print(df)
        res = {}
        market_price = [0]
        predict_price = [0]
        date = ['--/--/--']
        principle = [10000]
        position = [0]
        cost = [0]
        profit = [0]
        start, n = min(data['index'].values), data.shape[0]
        print("---------n:{}, start:{}".format(n, start))

        for index in range(start - 1, n + start - 1):
            day = data.at[index, 'Date']
            curr_price = data.at[index, 'market_price']
            pred_price = pre.loc[pre['forecast_time'] == day, 't+0'].values
            if pred_price.size == 0:
                continue
            pred_price = pred_price[0]
            print("date:{0}----cur:{1}-----pred:{2}".format(day, curr_price, pred_price))
            # execute buy
            if position[-1] == 0:
                if pred_price > curr_price:
                    num = principle[-1] / curr_price
                    position.append(num)
                    principle.append(0)
                    cost.append(curr_price)
                    print("num:{0}".format(num))
                else:
                    position.append(0)
                    principle.append(principle[-1])
                    cost.append(0)
            else:  # execute sell
                if pred_price < curr_price or round(curr_price / cost[-1] - 1, 4) > 0.05 or round(
                        curr_price / cost[-1] - 1, 4) < -0.03:
                    principle.append(position[-1] * curr_price)
                    position.append(0)
                    cost.append(0)
                else:
                    position.append(position[-1])
                    principle.append(0)
                    cost.append(cost[-1])
            date.append(day)
            market_price.append(curr_price)
            predict_price.append(pred_price)
            profit.append(round((cost[-1] * position[-1] + principle[-1]) / principle[0] - 1, 4))

        res['date'] = date
        res['market_price'] = market_price
        res['predict_price'] = predict_price
        res['principle'] = principle
        res['position'] = position
        res['cost'] = cost
        res['profit'] = profit
        res = pd.DataFrame(res)
        res.to_csv('backtesting.csv')
        #print(res)
        
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
    # inputs = tft_strategy.get_input_data()
    # predict_result = tft_strategy.predict_batch(inputs)
    # print(tft_strategy.formatter.format_predictions(predict_result))
    tft_strategy.predict_all()
    tft_strategy.execute_strategy()
