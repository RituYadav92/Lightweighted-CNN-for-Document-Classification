#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
from train.helper import data_helpers
from tensorflow.contrib import learn
import csv
from sklearn import metrics
import yaml
import sys

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
    

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

# Parameters
# ==================================================

# Eval Parameters
tf.compat.v1.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.compat.v1.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.compat.v1.flags.DEFINE_string("test_dir", "", "Test directory containing test data")
tf.compat.v1.flags.DEFINE_boolean("out_test", False, "gives output to the test data")
# Misc Parameters
tf.compat.v1.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.compat.v1.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.compat.v1.flags.FLAGS

YML_PATH = os.path.join(ROOT_DIR, "train/helper/config.yml")
print("***********************YML_PATH", YML_PATH)

with open(YML_PATH, 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

datasets = None

# CHANGE THIS: Load data. Load your own data here
dataset_name = cfg["datasets"]["default"]
print('dataset_name: ', dataset_name)

read_file = 'data/test_data/'

if FLAGS.out_test:
    # x_raw = ["sonntagsfrage bundestagswahl infratest dimapard cducsu 37 spd 25 grü 11 afd 10 lin 8 fdp 4", "die mehrheit der muslime beherrscht kunstvoll ihr rhetorisches spiel sie stellen die muslime als die ewigen opfer da"]
    x_raw = list(open(read_file, "r").readlines())
    print(x_raw)

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("size of x_test:", len(x_test))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.compat.v1.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.compat.v1.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.compat.v1.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.compat.v1.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        scores = graph.get_operation_by_name("output/scores").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
        all_probabilities = None

        for x_dev_batch in batches:
            batch_predictions_scores = sess.run([predictions, scores], {input_x: x_dev_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions_scores[0]])
            probabilities = softmax(batch_predictions_scores[1])
            if all_probabilities is not None:
                all_probabilities = np.concatenate([all_probabilities, probabilities])
            else:
                all_probabilities = probabilities

print(all_predictions)
print(all_probabilities)

# Save the predictions to a csv
predictions_human_readable = np.column_stack((np.array(x_raw),
                                             [int(prediction) for prediction in all_predictions],
                                             [ "{}".format(probability) for probability in all_probabilities]))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
   csv.writer(f).writerows(predictions_human_readable)
