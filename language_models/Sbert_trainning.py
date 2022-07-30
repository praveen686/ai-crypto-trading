import pandas as pd
import string
import re
import os
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses, datasets
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import os
import gzip
import csv
punctuation_string = string.punctuation
filename = 'NYTimes_all_headline.txt'
dataset = []
def callback(value, a, b):
  print('callback:', value, a, b)
  if math.isnan(value):
    raise optuna.exceptions.TrialPruned()

with open(filename, 'r') as fin:
  lines = fin.readlines()
  for line in lines:
    line = line.strip()
    line = re.sub('[{}]'.format(punctuation_string),"",line)
    dataset.append(line)

import nltk
nltk.download('punkt')
model_name = 'bert-base-uncased'
# Training parameters
model_name = 'bert-base-uncased'
train_batch_size = 8
num_epochs = 1
max_seq_length = 75
# Save path to store our model
model_save_path = 'output/training_stsb_tsdae-{}-{}-{}'.format(model_name, train_batch_size, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

# Check if dataset exsist. If not, download and extract  it
sts_dataset_path = 'data/stsbenchmark.tsv.gz'
# We use 1 Million sentences from Wikipedia to train our model
wikipedia_dataset_path = 'data/wiki1m_for_simcse.txt'
if not os.path.exists(wikipedia_dataset_path):
  util.http_get('https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt', wikipedia_dataset_path)

from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
# Read STSbenchmark dataset and use it as development set
logging.info("Read STSbenchmark dev dataset")
dev_samples = []
test_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
  reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
  for row in reader:
    score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
    if row['split'] == 'dev':
      dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
    elif row['split'] == 'test':
      test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='sts-test')

# Define your sentence transformer model using CLS pooling
model_name = 'bert-base-uncased'
word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'cls')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model],device='cuda')

# Define a list with sentences (1k - 100k sentences)
#train_sentences = ["Your set of sentences",
#                   "Model will automatically add the noise",
#                   "And re-construct it",
#                   "You should provide at least 1k sentences"]
train_sentences = dataset
# Create the special denoising dataset that adds noise on-the-fly
train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)

# DataLoader to batch your data
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Use the denoising auto-encoder loss
train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=model_name, tie_encoder_decoder=True)
logging.info("Training sentences: {}".format(len(train_sentences)))
logging.info("Performance before training")
print("Performance before training")
dev_evaluator(model)
# Call the fit method
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=10,
    evaluator=dev_evaluator,
    evaluation_steps=100,
    weight_decay=0,
    callback = callback,
    scheduler='constantlr',
    optimizer_params={'lr': 3e-5},
    show_progress_bar=True,
    use_amp=True
)

model.save(model_save_path)
test_evaluator(model, output_path=model_save_path)
