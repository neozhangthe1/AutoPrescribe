from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.models.rnn.translate import seq2seq_model
from utils.data import get_model_path, load
from utils.eval import Evaluator
import copy

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.


class Seq2Seq(object):

    def __init__(self,
                 lr=0.5,
                 lr_decay=0.99,
                 max_gradient_norm=5.0,
                 batch_size=20,
                 size=256,
                 num_layers=2,
                 num_samples=512,
                 steps_per_checkpoint=200,
                 use_fp16=False,
                 train_dir="seq2seq/"):
        self.train_set = []
        self.dev_set = []
        self.input_vocab = None
        self.ouput_vocab = None
        self.input_vocab_size = 0
        self.output_vocab_size = 0
        self.session = tf.InteractiveSession()
        self.lr = lr
        self.lr_decay = lr_decay
        self.max_gradient_norm = max_gradient_norm,
        self.batch_size = batch_size
        self.size = size
        self.num_samples = num_samples
        self.num_layers = num_layers
        self.steps_per_checkpoint = steps_per_checkpoint
        self.use_fp16 = use_fp16
        self.train_dir = get_model_path(train_dir)
        self.buckets = [(5, 10), (10, 15), (15, 25), (20, 50), (25, 100)]
        
        self.model = None
        self.global_step = None
        self.learning_rate = None
        self.learning_rate_decay_op = None
        
        self.PAD_ID = 4127
        self.GO_ID = 4128
        self.EOS_ID = 4129
        self.UNK_ID = 4130

    def __del__(self):
        self.session.close()

    def load_data(self, input_vocab, output_vocab, train_set, test_set):
        self.input_vocab = input_vocab
        self.ouput_vocab = output_vocab
        self.input_vocab_size = len(input_vocab)
        self.output_vocab_size = len(output_vocab) + 4
        self.train_set = self.read_data(train_set)
        self.dev_set = self.read_data(test_set, 100)

        self.PAD_ID = self.output_vocab_size - 1
        self.GO_ID = self.PAD_ID - 1
        self.EOS_ID = self.PAD_ID - 2
        self.UNK_ID = self.PAD_ID - 3

    def read_data(self, train_set, max_size=None):
        data_set = [[] for _ in self.buckets]

        for i, pair in enumerate(train_set):
            if max_size is not None:
                if i > max_size:
                    break
            source_ids = [x for x in pair[0]]
            target_ids = [x for x in pair[1]]
            for bucket_id, (source_size, target_size) in enumerate(self.buckets):
                if len(source_ids) < source_size and len(target_ids) < target_size:
                    data_set[bucket_id].append([source_ids, target_ids])
                    break
        return data_set

    def create_model(self, forward_only):
        """Create translation model and initialize or load parameters in session."""
        dtype = tf.float16 if self.use_fp16 else tf.float32
        print("Creating %d layers of %d units." % (self.num_layers, self.size))

        model = seq2seq_model.Seq2SeqModel(
            self.input_vocab_size,
            self.output_vocab_size,
            self.buckets,
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
            model.saver.restore(self.session, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            self.session.run(tf.initialize_all_variables())
        self.model = model
        return model

    def load(self):
        self.model = self.create_model(True)
        
    def get_batch(self, data, bucket_id):
        """Get a random batch of data from the specified bucket, prepare for step.
        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.
        Args:
          data: a tuple of size len(self.buckets) in which each element contains
            lists of pairs of input and output data that we use to create a batch.
          bucket_id: integer, which bucket to get the batch for.
        Returns:
          The triple (encoder_inputs, decoder_inputs, target_weights) for
          the constructed batch that has the proper format to call step(...) later.
        """
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in range(min(self.batch_size, len(data[bucket_id]))):
            encoder_input, decoder_input = copy.deepcopy(random.choice(data[bucket_id]))
            random.shuffle(encoder_input)
            random.shuffle(decoder_input)

            # Encoder inputs are padded and then reversed.
            encoder_pad = [self.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(encoder_input + encoder_pad))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 2
            decoder_inputs.append([self.GO_ID] + decoder_input + [self.EOS_ID] +
                                  [self.PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in range(encoder_size):
            batch_encoder_inputs.append(
                    np.array([encoder_inputs[batch_idx][length_idx]
                              for batch_idx in range(self.batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in range(decoder_size):
            batch_decoder_inputs.append(
                    np.array([decoder_inputs[batch_idx][length_idx]
                              for batch_idx in range(self.batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in range(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                target = None
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == self.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights, encoder_inputs, decoder_inputs

    def fit(self):
        # Create model.
        model = self.model
        train_bucket_sizes = [len(self.train_set[b]) for b in range(len(self.buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in trainself.buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights, encoder_inputs, decoder_inputs = self.get_batch(self.train_set, bucket_id)
            _, step_loss, _ = model.step(self.session, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / self.steps_per_checkpoint
            loss += step_loss / self.steps_per_checkpoint
            current_step += 1

            print(current_step, self.buckets[bucket_id], loss, step_loss)

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % self.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print("global step %d learning rate %.4f step-time %.2f perplexity "
                      "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    self.session.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                print(previous_losses[-10:])
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(self.train_dir, "mimic.ckpt")
                model.saver.save(self.session, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                for bucket_id in range(len(self.buckets)):
                    if len(self.dev_set[bucket_id]) == 0:
                        print("eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights, encoder_inputs, decoder_inputs = self.get_batch(self.dev_set, bucket_id)
                    _, eval_loss, _ = model.step(self.session, encoder_inputs, decoder_inputs,
                                                 target_weights, bucket_id, True)
                    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                    print("eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                    to_predict = self.dev_set[bucket_id][0]
                    outputs = self.predict(to_predict[0])
                    precision, recall = Evaluator.get_result(to_predict[1], outputs)
                    print("inputs: ", to_predict[0])
                    print("target: ", to_predict[1])
                    print("output: ", outputs)
                    print("pre", precision, "rec", recall)
                sys.stdout.flush()

    def predict(self, inputs):
        bucket_id = len(self.buckets) - 1
        for i, bucket in enumerate(self.buckets):
            if bucket[0] >= len(inputs):
                bucket_id = i
                break

        encoder_inputs, decoder_inputs, target_weights, encoder_inputs, decoder_inputs = self.get_batch({
            bucket_id: [(inputs, [])]
        }, bucket_id)

        _, _, output_logits = self.model.step(
            self.session,
            encoder_inputs,
            decoder_inputs,
            target_weights,
            bucket_id,
            True
        )

        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        return outputs

    def decode(self, test_set):
        # Create model and load parameters.
        model = self.create_model(True)
        model.batch_size = 1  # We decode one sentence at a time.

        # Load vocabularies.
        input_vocab = load("diag_vocab.pkl")
        output_vocab = load("drug_vocab.pkl")

        output_id_to_token = {}
        for token in output_vocab:
            output_id_to_token[output_vocab[token]] = token

        # Decode from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        for pair in test_set:
            # Get token-ids for the input sentence.
            token_ids = [input_vocab[token] for token in pair[0]]
            # Which bucket does it belong to?
            bucket_id = len(self.buckets) - 1
            for i, bucket in enumerate(self.buckets):
                if bucket[0] >= len(token_ids):
                    bucket_id = i
                    break
            else:
                logging.warning("Sentence truncated: %s", pair[0])

            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, target_weights, encoder_inputs, decoder_inputs = self.get_batch(
                    {bucket_id: [(token_ids, [])]}, bucket_id
            )
            # Get output logits for the sentence.
            _, _, output_logits = model.step(self.session, encoder_inputs, decoder_inputs,
                                             target_weights, bucket_id, True)
            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            # If there is an EOS symbol in outputs, cut them at that point.
            if self.EOS_ID in outputs:
                outputs = outputs[:outputs.index(self.EOS_ID)]
            # Print out French sentence corresponding to outputs.
            print(" ".join([output_id_to_token[output] for output in outputs]))
            print("> ", end="")
            sys.stdout.flush()


def train():
    input_vocab = load("diag_vocab.pkl")
    output_vocab = load("drug_vocab.pkl")
    test_set = load("mimic_episodes_index_test.pkl")
    train_set = load("mimic_episodes_index_train.pkl")
    seq2seq = Seq2Seq()
    seq2seq.load_data(input_vocab, output_vocab, train_set, test_set)
    model = seq2seq.create_model(False)
    seq2seq.fit()


