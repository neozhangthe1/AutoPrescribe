from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import os
import pickle
import random
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.models.rnn.translate import data_utils
from tensorflow.models.rnn.translate import seq2seq_model


# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (15, 25), (20, 50), (25, 100)]


class Seq2Seq(object):

    def __init__(self,
                 lr=0.5,
                 lr_decay=0.99,
                 max_gradient_norm=5.0,
                 batch_size=10,
                 size=1024,
                 num_layers=3,
                 steps_per_checkpoint=200,
                 use_fp16=False,
                 train_dir="./build/"):
        self.train_set = []
        self.dev_set = []
        self.input_vocab_size = 0
        self.output_vocab_size = 0
        self.session = None
        self.lr = lr
        self.lr_decay = lr_decay
        self.max_gradient_norm = max_gradient_norm,
        self.batch_size = batch_size
        self.size = size
        self.num_layers = num_layers
        self.steps_per_checkpoint = steps_per_checkpoint
        self.use_fp16 = use_fp16
        self.train_dir = train_dir

    def load_data(self, input_vocab_size, output_vocab_size, train_set, test_set):
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.train_set = self.read_data(train_set)
        self.dev_set = self.read_data(test_set, 100)

    @staticmethod
    def read_data(train_set, max_size=None):
        data_set = [[] for _ in _buckets]

        for i, pair in enumerate(train_set):
            if max_size is not None:
                if i > max_size:
                    break
            source_ids = [x for x in pair[0]]
            target_ids = [x for x in pair[1]]
            target_ids.append(4127)
            for bucket_id, (source_size, target_size) in enumerate(_buckets):
                if len(source_ids) < source_size and len(target_ids) < target_size:
                    data_set[bucket_id].append([source_ids, target_ids])
                    break
        return data_set

    def create_model(self, session, forward_only):
        """Create translation model and initialize or load parameters in session."""
        self.session = session
        dtype = tf.float16 if self.use_fp16 else tf.float32
        model = seq2seq_model.Seq2SeqModel(
            self.input_vocab_size,
            self.output_vocab_size,
            _buckets,
            self.size,
            self.num_layers,
            self.max_gradient_norm,
            self.batch_size,
            self.lr,
            self.lr_decay,
            forward_only=forward_only,
            dtype=dtype)
        ckpt = tf.train.get_checkpoint_state(self.train_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            session.run(tf.initialize_all_variables())
        return model

    def fit(self):
        with tf.Session() as sess:
            # Create model.
            print("Creating %d layers of %d units." % (self.num_layers, self.size))
            model = self.create_model(sess, False)

            train_bucket_sizes = [len(self.train_set[b]) for b in range(len(_buckets))]
            train_total_size = float(sum(train_bucket_sizes))

            train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                                   for i in range(len(train_bucket_sizes))]

            # This is the training loop.
            step_time, loss = 0.0, 0.0
            current_step = 0
            previous_losses = []
            while True:
                # Choose a bucket according to data distribution. We pick a random number
                # in [0, 1] and use the corresponding interval in train_buckets_scale.
                random_number_01 = np.random.random_sample()
                bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])

                # Get a batch and make a step.
                start_time = time.time()
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(self.train_set, bucket_id)
                _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
                step_time += (time.time() - start_time) / self.steps_per_checkpoint
                loss += step_loss / self.steps_per_checkpoint
                current_step += 1

                print(current_step, _buckets[bucket_id], loss, step_loss)

                # Once in a while, we save checkpoint, print statistics, and run evals.
                if current_step % self.steps_per_checkpoint == 0:
                    # Print statistics for the previous epoch.
                    perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                    print("global step %d learning rate %.4f step-time %.2f perplexity "
                          "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                    step_time, perplexity))
                    # Decrease learning rate if no improvement was seen over last 3 times.
                    if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                        sess.run(model.learning_rate_decay_op)
                    previous_losses.append(loss)
                    print(previous_losses[-10:])
                    # Save checkpoint and zero timer and loss.
                    checkpoint_path = os.path.join(self.train_dir, "mimic.ckpt")
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                    step_time, loss = 0.0, 0.0
                    # Run evals on development set and print their perplexity.
                    for bucket_id in range(len(_buckets)):
                        if len(self.dev_set[bucket_id]) == 0:
                            print("eval: empty bucket %d" % (bucket_id))
                            continue
                        encoder_inputs, decoder_inputs, target_weights = model.get_batch(self.dev_set, bucket_id)
                        _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                     target_weights, bucket_id, True)
                        eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                        print("eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                    sys.stdout.flush()

    def decode(self):
        with tf.Session() as sess:
            # Create model and load parameters.
            model = self.create_model(sess, True)
            model.batch_size = 1  # We decode one sentence at a time.

            # Load vocabularies.
            input_vocab = pickle.load(open("diag_vocab.pkl", "rb"))
            output_vocab = pickle.load(open("drug_vocab.pkl", "rb"))

            output_id_to_token = {}
            for token in output_vocab:
                output_id_to_token[output_vocab[token]] = token

            test_set = pickle.load(open("mimic_episodes_test.pkl", "rb"))

            # Decode from standard input.
            sys.stdout.write("> ")
            sys.stdout.flush()
            for pair in test_set:
                # Get token-ids for the input sentence.
                token_ids = [input_vocab[token] for token in pair[0]]
                # Which bucket does it belong to?
                bucket_id = len(_buckets) - 1
                for i, bucket in enumerate(_buckets):
                    if bucket[0] >= len(token_ids):
                        bucket_id = i
                        break
                else:
                    logging.warning("Sentence truncated: %s", pair[0])

                # Get a 1-element batch to feed the sentence to the model.
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                        {bucket_id: [(token_ids, [])]}, bucket_id)
                # Get output logits for the sentence.
                _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                                 target_weights, bucket_id, True)
                # This is a greedy decoder - outputs are just argmaxes of output_logits.
                outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
                # If there is an EOS symbol in outputs, cut them at that point.
                if data_utils.EOS_ID in outputs:
                    outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                # Print out French sentence corresponding to outputs.
                print(" ".join([output_id_to_token[output] for output in outputs]))
                print("> ", end="")
                sys.stdout.flush()

