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
tf.compat.v1.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.compat.v1.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.compat.v1.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.compat.v1.flags.FLAGS

YML_PATH = os.path.join(ROOT_DIR, "train/helper/config.yml")
print("***********************YML_PATH", YML_PATH)

with open(YML_PATH, 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

datasets = None

# CHANGE THIS: if you want to load your own data.
dataset_name = cfg["datasets"]["default"]
print('dataset_name: ', dataset_name)
if FLAGS.eval_train:
    if dataset_name == "tobacco":
        print("LOading data !")
        datasets = data_helpers.get_datasets_tobacco()
    elif dataset_name == "localdata":
        datasets = data_helpers.get_datasets_localdata(container_path=cfg["datasets"][dataset_name]["container_path"],
                                                       categories=cfg["datasets"][dataset_name]["categories"],
                                                       shuffle=cfg["datasets"][dataset_name]["shuffle"],
                                                       random_state=cfg["datasets"][dataset_name]["random_state"])
        
    x_raw, y_test = data_helpers.load_data_labels(datasets)
    y_test = np.argmax(y_test, axis=1)
    print("Total number of test examples: {}".format(len(y_test)))
else:
    print("Flow shouldn't be here.")

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y_test)))
x_test = x_test[shuffle_indices]
y_test = y_test[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(0.2 * float(len(y_test)))
x_train, x_dev = x_test[:dev_sample_index], x_test[dev_sample_index:]
y_train, y_dev = y_test[:dev_sample_index], y_test[dev_sample_index:]

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
print("size of x_dev, y_dev:", len(x_dev), len(y_dev))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.compat.v1.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
g=tf.get_default_graph()
with graph.as_default():
    tf.compat.v1.contrib.quantize.create_eval_graph(input_graph=g)
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
        batches = data_helpers.batch_iter(list(x_dev), FLAGS.batch_size, 1, shuffle=False)

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

# Print accuracy if y_dev is defined
if y_dev is not None:
    correct_predictions = float(sum(all_predictions == y_dev))
    print("Total number of test examples: {}".format(len(y_dev)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_dev))))
    print(metrics.classification_report(y_dev, all_predictions, target_names=datasets['target_names']))
    print(metrics.confusion_matrix(y_dev, all_predictions))

# Save the evaluation to a csv
# predictions_human_readable = np.column_stack((np.array(x_raw),
#                                              [int(prediction) for prediction in all_predictions],
#                                              [ "{}".format(probability) for probability in all_probabilities]))
# out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
# print("Saving evaluation to {0}".format(out_path))
# with open(out_path, 'w') as f:
#    csv.writer(f).writerows(predictions_human_readable)
