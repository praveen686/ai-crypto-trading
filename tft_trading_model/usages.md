# Environments setup
conda create -n tft_trading python=3.7
pip3 install -r requirements.txt

# Download and construct datesets
python3 -m script_download_data bitcoin ../../tft_script

# Run model training (GPU required)
python3 -m script_train_fixed_params bitcoin ../../tft_script yes

# Hyperparameters tuning
Modify data_formatters/bitcoin.py line 190

# Run backtesting
python tft_trading_strategy.py

# References
The bitcoin-related data is from Nasdaq Data Link API: https://data.nasdaq.com/
The TFT model implementation refers to the code of this work: https://arxiv.org/abs/1912.09363


