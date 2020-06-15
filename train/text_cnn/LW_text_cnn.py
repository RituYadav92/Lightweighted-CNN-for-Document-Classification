import tensorflow as tf
import numpy as np

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0001):

        # Placeholders for input, output and dropout
        self.input_x = tf.compat.v1.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.compat.v1.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")
        self.learning_rate = tf.compat.v1.placeholder(tf.float32)
        
        
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random.uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                
                #Batch Normalzation
                conv_bn1 = tf.layers.batch_normalization(self.embedded_chars_expanded, momentum=0.9)

                #depthwise Convolution Layer 
                
                #+Dialated Conv atrous_conv2d
                if filter_size==5:
                    filter_shape = [3, embedding_size, 1, 1]
                    W = tf.Variable(tf.random.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv2 = tf.nn.atrous_conv2d(
                        #self.embedded_chars_expanded,
                        conv_bn1,
                        W,
                        rate=2,
                        padding="SAME",
                        name="conv2"
                        )
                    
                    print(conv2.shape)
                    
                    #Seperable Conv/depth
                    filter_shape3 = [3, embedding_size, 1, 1]
                    W3 = tf.Variable(tf.random.truncated_normal(filter_shape3, stddev=0.1), name="W3")
                    conv = tf.nn.conv2d(
                        conv2,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    
                    print(conv.shape)
                    
                    ksize_1= [1, sequence_length - 2, 1, 1]
                    
                
                else:                    
                    filter_shape2 = [filter_size, embedding_size, 1, 1]
                    W2 = tf.Variable(tf.random.truncated_normal(filter_shape2, stddev=0.1), name="W2")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        #self.embedded_chars_expanded,
                        conv_bn1,
                        W2,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    print(conv.shape) 
                    ksize_1= [1, sequence_length - filter_size + 1, 1, 1]
                
                                
                #Pointwise Convolution Layer
                filter_shape1 = [1, 1, 1, num_filters]
                W1 = tf.Variable(tf.random.truncated_normal(filter_shape1, stddev=0.1), name="W1")
                conv1 = tf.nn.conv2d(
                    conv,
                    W1,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv1")                
                                                
                
                #Batch Normalzation                
                conv_bn2 = tf.layers.batch_normalization(conv1, momentum=0.9)
                
                # Apply nonlinearity
                h = tf.nn.leaky_relu(tf.nn.bias_add(conv_bn2, b),0.1 ,name="leakyRelu")
                
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool2d(
                    h,
                    ksize=ksize_1,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
              
        
        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.compat.v1.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.compat.v1.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")