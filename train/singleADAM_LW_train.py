#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from helper import data_helpers
from text_cnn.LW_text_cnn import TextCNN
from tensorflow.contrib import learn
import yaml
import math
import sys

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
    
# Parameters
# ==================================================

# Data loading params
tf.compat.v1.flags.DEFINE_float("dev_sample_percentage", 0.2, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.compat.v1.flags.DEFINE_boolean("enable_word_embeddings", False, "Enable/disable the word embedding (default: True)")
tf.compat.v1.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 128)")
tf.compat.v1.flags.DEFINE_string("filter_sizes", "2,3,5", "Comma-separated filter sizes (default: '2,3,5')")#3,4,5 ##2,4,5
tf.compat.v1.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")#128
tf.compat.v1.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.compat.v1.flags.DEFINE_float("l2_reg_lambda", 0.0001, "L2 regularization lambda (default: 0.0)")#0.001

# Training parameters
tf.compat.v1.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.compat.v1.flags.DEFINE_integer("num_epochs",20, "Number of training epochs (default: 200)")
tf.compat.v1.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.compat.v1.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.compat.v1.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.compat.v1.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.compat.v1.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.compat.v1.flags.DEFINE_float("decay_coefficient", 2.5, "Decay coefficient (default: 2.5)")

FLAGS = tf.compat.v1.flags.FLAGS

YML_PATH = os.path.join(ROOT_DIR, "train/helper/config.yml")
print("***********************YML_PATH", YML_PATH)

with open(YML_PATH, 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

dataset_name = cfg["datasets"]["default"]
if FLAGS.enable_word_embeddings and cfg['word_embeddings']['default'] is not None:
    embedding_name = cfg['word_embeddings']['default']
    embedding_dimension = cfg['word_embeddings'][embedding_name]['dimension']
else:
    embedding_dimension = FLAGS.embedding_dim

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
datasets = None
if dataset_name == "tobacco":
    data_path = os.path.join(ROOT_DIR, cfg["datasets"][dataset_name]['parent_dir'] +'/')
    print("*************data_path", data_path)    
    datasets = data_helpers.get_datasets_tobacco(data_path)

elif dataset_name == "localdata":
    datasets = data_helpers.get_datasets_localdata(container_path=cfg["datasets"][dataset_name]["container_path"],
                                                     categories=cfg["datasets"][dataset_name]["categories"],
                                                     shuffle=cfg["datasets"][dataset_name]["shuffle"],
                                                     random_state=cfg["datasets"][dataset_name]["random_state"])

x_text, y = data_helpers.load_data_labels(datasets)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.compat.v1.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.compat.v1.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=embedding_dimension,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
        
        global_step = tf.compat.v1.Variable(0, name="global_step", trainable=False)
        optimizer = tf.compat.v1.train.AdamOptimizer(cnn.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        
        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.compat.v1.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.compat.v1.summary.scalar("{}/grad/sparsity".format(v.name), 
                                                               tf.compat.v1.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.compat.v1.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.compat.v1.summary.scalar("loss", cnn.loss)
        acc_summary = tf.compat.v1.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.compat.v1.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.compat.v1.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.compat.v1.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.compat.v1.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.compat.v1.global_variables_initializer())
        if FLAGS.enable_word_embeddings and cfg['word_embeddings']['default'] is not None:
            vocabulary = vocab_processor.vocabulary_
            initW = None
            if embedding_name == 'word2vec':
                # load embedding vectors from the word2vec
                print("Load word2vec file {}".format(cfg['word_embeddings']['word2vec']['path']))
                initW = data_helpers.load_embedding_vectors_word2vec(vocabulary,
                                                                     cfg['word_embeddings']['word2vec']['path'],
                                                                     cfg['word_embeddings']['word2vec']['binary'])
                print("word2vec file has been loaded")
            elif embedding_name == 'glove':
                # load embedding vectors from the glove
                print("Load glove file {}".format(cfg['word_embeddings']['glove']['path']))
                initW = data_helpers.load_embedding_vectors_glove(vocabulary,
                                                                  cfg['word_embeddings']['glove']['path'],
                                                                  embedding_dimension)
                print("glove file has been loaded\n")
            sess.run(cnn.W.assign(initW))

        def train_step(x_batch, y_batch, learning_rate):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,                
              cnn.learning_rate: learning_rate                           
                
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}, learning_rate {:g}"
                  .format(time_str, step, loss, accuracy, learning_rate))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
              
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
                

        print('Trainning input set: x_train, y_train',len(list(zip(x_train, y_train))))
        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        #It uses dynamic learning rate with a high value at the beginning to speed up the training
        max_learning_rate = 0.004
        min_learning_rate = 0.0001        
        decay_speed = FLAGS.decay_coefficient*len(y_train)/FLAGS.batch_size
        # Training loop. For each batch...
        counter = 0
        print("*********Trainable PARAMETERS***********",np.sum([np.prod(v.get_shape().as_list()) for v in 
                                                                 tf.compat.v1.trainable_variables()]))
        for batch in batches:
            learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-counter/decay_speed)
            counter += 1
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch, learning_rate)
            
            current_step = tf.compat.v1.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))