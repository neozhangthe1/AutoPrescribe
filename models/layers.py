import lasagne
from theano import sparse
import theano.tensor as T
import numpy as np
import theano

INF = 1e8

def log_softmax(x):
    x_max = x.max(axis = 1, keepdims = True)
    x -= x_max
    return x - T.log(T.sum(T.exp(x), axis = 1, keepdims = True))

def get_sigmoid_comb(x):
    """
        return: sigmoid(x), log(sigmoid(x)), log(1 - sigmoid(x))
    """
    x_max = T.maximum(x, 0.0)
    x_exp, zero_exp = T.exp(x - x_max), T.exp(0.0 - x_max)
    return x_exp / (x_exp + zero_exp), x - x_max - T.log(x_exp + zero_exp), 0.0 - x_max - T.log(x_exp + zero_exp)

class GRUCoverageTrainLayer(lasagne.layers.MergeLayer):
    def __init__(self, incoming, num_units,
                 resetgate=lasagne.layers.Gate(W_cell=None),
                 updategate=lasagne.layers.Gate(W_cell=None),
                 hidden_update=lasagne.layers.Gate(W_cell=None,
                                    nonlinearity=lasagne.nonlinearities.tanh),
                 hid_init=lasagne.init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=False,
                 mask_input=None,
                 only_return_final=False,
                 l_enc_feat=None,
                 l_enc_mask=None,
                 l_map=None,
                 l_output=None,
                 source_token_cnt=None,
                 target_token_cnt=None,
                 # extra_word_cnt=None,
                 W_emb=None,
                 # W_gen=lasagne.init.GlorotUniform(),
                 # W_copy=lasagne.init.GlorotUniform(),
                 # W_mode=lasagne.init.Normal(),
                 unk_index=None,
                 **kwargs):

        incomings = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if isinstance(hid_init, lasagne.layers.Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1

        incomings.append(l_enc_feat)
        self.enc_feat_index = len(incomings) - 1
        incomings.append(l_enc_mask)
        self.enc_mask_index = len(incomings) - 1
        # incomings.append(l_map)
        # self.map_index = len(incomings) - 1
        incomings.append(l_output)
        self.output_index = len(incomings) - 1

        for inc in incomings:
            print(inc)

        # Initialize parent layer
        super(GRUCoverageTrainLayer, self).__init__(incomings, **kwargs)

        self.learn_init = learn_init
        self.num_units = num_units
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final
        self.source_token_cnt = source_token_cnt
        self.target_token_cnt = target_token_cnt
        # self.extra_word_cnt = extra_word_cnt
        self.W_emb = W_emb
        self.unk_index = unk_index

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        # Input dimensionality is the output dimensionality of the input layer
        # num_inputs = np.prod(input_shape[2:])
        num_inputs = self.W_emb.get_value().shape[1]

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs + num_units + num_units, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in all parameters from gates
        (self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')

        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
         self.b_hidden_update, self.nonlinearity_hid) = add_gate_params(
             hidden_update, 'hidden_update')

        self.updategate = lasagne.layers.Gate(W_in = self.W_in_to_updategate, W_hid = self.W_hid_to_updategate, W_cell = None, b = self.b_updategate, nonlinearity = self.nonlinearity_updategate)
        self.resetgate = lasagne.layers.Gate(W_in = self.W_in_to_resetgate, W_hid = self.W_hid_to_resetgate, W_cell = None, b = self.b_resetgate, nonlinearity = self.nonlinearity_resetgate)
        self.hidden_update = lasagne.layers.Gate(W_in = self.W_in_to_hidden_update, W_hid = self.W_hid_to_hidden_update, W_cell = None, b = self.b_hidden_update, nonlinearity = self.nonlinearity_hid)

        # Initialize hidden state
        if isinstance(hid_init, lasagne.layers.Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

        # self.W_gen = self.add_param(W_gen, (self.num_units, self.target_token_cnt), name = "W_gen")
        # self.W_copy = self.add_param(W_copy, (self.num_units, self.num_units), name = "W_copy")
        # self.W_mode = self.add_param(W_mode, (self.num_units, ), name = "W_mode")

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, inputs, **kwargs):
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        if self.mask_incoming_index >= 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index >= 0:
            hid_init = inputs[self.hid_init_incoming_index]

        enc_feat = inputs[self.enc_feat_index]
        enc_mask = inputs[self.enc_mask_index]
        # map = inputs[self.map_index] # (batch, enc_len, vocab + extra)
        output = inputs[self.output_index] # (batch, dec_len)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0)
        output = output.dimshuffle(1, 0)

        seq_len, num_batch = input.shape

        # Stack input weight matrices into a (num_inputs, 3*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate, self.W_in_to_updategate,
             self.W_in_to_hidden_update], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1)

        # Stack gate biases into a (3*num_units) vector
        b_stacked = T.concatenate(
            [self.b_resetgate, self.b_updategate,
             self.b_hidden_update], axis=0)

        # When theano.scan calls step, input_n will be (n_batch, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        def step(input_n, output_n, hid_previous, copy_hid_previous, prob_previous, *args):
            """
            input_n: (batch, ); each entry is the index; no extra vocabulary.
            output_n: (batch, ); each entry is the index; with extra vocabulary.
            hid_previous: (batch, units).
            prob_previous: (batch, vocab + extra).
            copy_hid_previous: (batch, units); a vector that combines the hidden states using the copy probabilities at the last time step.
            """
            input_emb = self.W_emb[input_n]

            # enc_feat: (batch, enc_len, units), hid_previous: (batch, units)
            att = T.batched_dot(enc_feat, hid_previous) # (batch, enc_len)
            att = T.nnet.softmax(att) * enc_mask # (batch, enc_len)
            att = att / (T.sum(att, axis = 1, keepdims = True) + 1e-8) # (batch, enc_len)
            att = T.batched_dot(att, enc_feat) # (batch, units)
            input_n = T.concatenate([input_emb, copy_hid_previous, att], axis = 1)

            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
            hid_input = T.dot(hid_previous, W_hid_stacked)

            if self.grad_clipping:
                input_n = theano.gradient.grad_clip(
                    input_n, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)

            if not self.precompute_input:
                # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            # Reset and update gates
            resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)

            # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
            hidden_update_in = slice_w(input_n, 2)
            hidden_update_hid = slice_w(hid_input, 2)
            hidden_update = hidden_update_in + resetgate*hidden_update_hid
            if self.grad_clipping:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
            hidden_update = self.nonlinearity_hid(hidden_update)

            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid = (1 - updategate)*hid_previous + updategate*hidden_update # (batch, units)

            # gen_score = T.dot(hid, self.W_gen)
            # gen_log_probs = log_softmax(gen_score)
            # copy_score = T.batched_dot(T.tanh(T.dot(enc_feat, self.W_copy)), hid)
            # copy_score -= copy_score.max(axis = 1, keepdims = True)
            # copy_score_no_map = T.exp(copy_score)
            # copy_score_map = copy_score.dimshuffle(0, 1, 'x') * map - INF * (1 - map) # (batch, enc_len, vocab + extra)
            # copy_score_max = copy_score_map.max(axis = 1) # (batch, vocab + extra)
            # copy_score = T.exp(copy_score_map - copy_score_max.dimshuffle(0, 'x', 1)).sum(axis = 1) # (batch, vocab + extra)
            # copy_score = copy_score_max + T.log(copy_score)
            # copy_log_probs = log_softmax(copy_score) # (batch, vocab + extra)
            # copy_mask = map.max(axis = 1) # (batch, vocab + extra)
            #
            # output_n = T.extra_ops.to_one_hot(output_n, self.word_cnt + self.extra_word_cnt) # (batch, vocab + extra)
            # output_n = T.batched_dot(output_n, map.dimshuffle(0, 2, 1)) # (batch, enc_len)
            # copy_probs = copy_score_no_map * output_n
            # copy_probs = copy_probs / T.maximum(T.sum(copy_probs, axis = 1, keepdims = True), 1e-8) # (batch, enc_len)
            # copy_hid = T.batched_dot(copy_probs, enc_feat) # (batch, units)
            #
            # copy_weight_sigmoid, copy_weight_log_sigmoid, copy_weight_log_neg_sigmoid = get_sigmoid_comb(T.dot(hid, self.W_mode))
            # copy_weight_sigmoid = copy_weight_sigmoid.dimshuffle(0, 'x')
            # copy_weight_log_sigmoid = copy_weight_log_sigmoid.dimshuffle(0, 'x')
            # copy_weight_log_neg_sigmoid = copy_weight_log_neg_sigmoid.dimshuffle(0, 'x')
            #
            # log_probs_max = T.maximum(copy_log_probs[:, : self.word_cnt], gen_log_probs) # (batch, vocab)
            # vocab_log_probs = T.log(T.exp(copy_log_probs[:, : self.word_cnt] - log_probs_max) * copy_weight_sigmoid + T.exp(gen_log_probs - log_probs_max) * (1 - copy_weight_sigmoid)) + log_probs_max
            # vocab_copy_mask = copy_mask[:, : self.word_cnt]
            # vocab_log_probs = vocab_copy_mask * vocab_log_probs + (1 - vocab_copy_mask) * (copy_weight_log_neg_sigmoid + gen_log_probs)
            #
            # extra_copy_mask = copy_mask[:, self.word_cnt :]
            # extra_log_probs = copy_log_probs[:, self.word_cnt :] + copy_weight_log_sigmoid
            # extra_log_probs = extra_copy_mask * extra_log_probs + (1 - extra_copy_mask) * (- INF)
            #
            # combined_probs = T.zeros_like(copy_log_probs)
            # combined_probs = T.set_subtensor(combined_probs[:, : self.word_cnt], vocab_log_probs)
            # combined_probs = T.set_subtensor(combined_probs[:, self.word_cnt :], extra_log_probs)
            # prob = combined_probs

            return hid #[hid, copy_hid, prob]

        def step_masked(input_n, output_n, mask_n, hid_previous, copy_hid_previous, prob_previous, *args):
            # [hid, copy_hid, prob] = step(input_n, output_n, hid_previous, copy_hid_previous, prob_previous, *args)
            #
            # # Skip over any input with mask 0 by copying the previous
            # # hidden state; proceed normally for any input with mask 1.
            # hid = T.switch(mask_n, hid, hid_previous)
            #
            # return [hid, copy_hid, prob]
            hid = step(input_n, hid_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            hid = T.switch(mask_n, hid, hid_previous)
            return hid

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, output, mask]
            step_fun = step_masked
        else:
            sequences = [input, output]
            step_fun = step

        if not isinstance(self.hid_init, lasagne.layers.Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)

        copy_hid_init = T.zeros((num_batch, self.num_units))

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        non_seqs += [enc_feat, enc_mask, self.W_emb]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            [hid_out, _, prob_out] = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init, copy_hid_init, None],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            [hid_out, _, prob_out] = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init, copy_hid_init, None],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            prob_out = prob_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            prob_out = prob_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                prob_out = prob_out[:, ::-1]

        return prob_out

class GRUCoverageTestLayer(lasagne.layers.MergeLayer):
    def __init__(self, num_units,
                 resetgate=lasagne.layers.Gate(W_cell=None),
                 updategate=lasagne.layers.Gate(W_cell=None),
                 hidden_update=lasagne.layers.Gate(W_cell=None,
                                    nonlinearity=lasagne.nonlinearities.tanh),
                 hid_init=lasagne.init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=False,
                 mask_input=None,
                 only_return_final=False,
                 l_enc_feat=None,
                 l_enc_mask=None,
                 # l_map=None,
                 source_token_cnt=None,
                 target_token_cnt=None,
                 W_emb=None,
                 # W_gen=lasagne.init.GlorotUniform(),
                 # W_copy=lasagne.init.GlorotUniform(),
                 # W_mode=lasagne.init.Normal(),
                 unk_index=None,
                 start_index=None,
                 gen_len=None,
                 **kwargs):

        incomings = []
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if isinstance(hid_init, lasagne.layers.Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1

        incomings.append(l_enc_feat)
        self.enc_feat_index = len(incomings) - 1
        incomings.append(l_enc_mask)
        self.enc_mask_index = len(incomings) - 1
        # incomings.append(l_map)
        # self.map_index = len(incomings) - 1

        # Initialize parent layer
        super(GRUCoverageTestLayer, self).__init__(incomings, **kwargs)

        self.learn_init = learn_init
        self.num_units = num_units
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final
        self.source_token_cnt = source_token_cnt
        self.target_token_cnt = target_token_cnt
        self.W_emb = W_emb
        self.unk_index = unk_index
        self.start_index = start_index
        self.gen_len = gen_len

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Input dimensionality is the output dimensionality of the input layer
        # num_inputs = np.prod(input_shape[2:])
        num_inputs = self.W_emb.get_value().shape[1]

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs + num_units + num_units, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in all parameters from gates
        (self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')

        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
         self.b_hidden_update, self.nonlinearity_hid) = add_gate_params(
             hidden_update, 'hidden_update')

        self.updategate = lasagne.layers.Gate(W_in = self.W_in_to_updategate, W_hid = self.W_hid_to_updategate, W_cell = None, b = self.b_updategate, nonlinearity = self.nonlinearity_updategate)
        self.resetgate = lasagne.layers.Gate(W_in = self.W_in_to_resetgate, W_hid = self.W_hid_to_resetgate, W_cell = None, b = self.b_resetgate, nonlinearity = self.nonlinearity_resetgate)
        self.hidden_update = lasagne.layers.Gate(W_in = self.W_in_to_hidden_update, W_hid = self.W_hid_to_hidden_update, W_cell = None, b = self.b_hidden_update, nonlinearity = self.nonlinearity_hid)

        # Initialize hidden state
        if isinstance(hid_init, lasagne.layers.Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

        # self.W_gen = self.add_param(W_gen, (self.num_units, self.target_token_cnt), name = "W_gen")
        # self.W_copy = self.add_param(W_copy, (self.num_units, self.num_units), name = "W_copy")
        # self.W_mode = self.add_param(W_mode, (self.num_units, ), name = "W_mode")

    def get_output_shape_for(self, input_shapes):
        return input_shapes[self.enc_feat_index][0], self.gen_len

    def get_output_for(self, inputs, **kwargs):
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        if self.mask_incoming_index >= 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index >= 0:
            hid_init = inputs[self.hid_init_incoming_index]

        enc_feat = inputs[self.enc_feat_index]
        enc_mask = inputs[self.enc_mask_index]
        # map = inputs[self.map_index] # (batch, enc_len, vocab + extra)

        num_batch = enc_feat.shape[0]

        # Stack input weight matrices into a (num_inputs, 3*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate, self.W_in_to_updategate,
             self.W_in_to_hidden_update], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1)

        # Stack gate biases into a (3*num_units) vector
        b_stacked = T.concatenate(
            [self.b_resetgate, self.b_updategate,
             self.b_hidden_update], axis=0)

        # When theano.scan calls step, input_n will be (n_batch, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        W_emb_comp = T.dot(T.ones((self.extra_word_cnt, 1)), self.W_emb[self.unk_index].dimshuffle('x', 0)) # (extra, emb)
        W_emb = T.concatenate([self.W_emb, W_emb_comp], axis = 0)

        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        def step(input_n, hid_previous, copy_hid_previous, *args):
            """
            input_n: (batch, ); each entry is the index; with extra vocabulary.
            hid_previous: (batch, units).
            prob_previous: (batch, vocab + extra).
            copy_hid_previous: (batch, units); a vector that combines the hidden states using the copy probabilities at the last time step.
            """
            input_emb = W_emb[input_n]

            # enc_feat: (batch, enc_len, units), hid_previous: (batch, units)
            att = T.batched_dot(enc_feat, hid_previous) # (batch, enc_len)
            att = T.nnet.softmax(att) * enc_mask # (batch, enc_len)
            att = att / (T.sum(att, axis = 1, keepdims = True) + 1e-8) # (batch, enc_len)
            att = T.batched_dot(att, enc_feat) # (batch, units)
            input_n = T.concatenate([input_emb, copy_hid_previous, att], axis = 1)

            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
            hid_input = T.dot(hid_previous, W_hid_stacked)

            if self.grad_clipping:
                input_n = theano.gradient.grad_clip(
                    input_n, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)

            if not self.precompute_input:
                # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            # Reset and update gates
            resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)

            # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
            hidden_update_in = slice_w(input_n, 2)
            hidden_update_hid = slice_w(hid_input, 2)
            hidden_update = hidden_update_in + resetgate*hidden_update_hid
            if self.grad_clipping:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
            hidden_update = self.nonlinearity_hid(hidden_update)

            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid = (1 - updategate)*hid_previous + updategate*hidden_update # (batch, units)

            # gen_score = T.dot(hid, self.W_gen)
            # gen_log_probs = log_softmax(gen_score)
            # copy_score = T.batched_dot(T.tanh(T.dot(enc_feat, self.W_copy)), hid)
            # copy_score -= copy_score.max(axis = 1, keepdims = True)
            # copy_score_no_map = T.exp(copy_score)
            # copy_score_map = copy_score.dimshuffle(0, 1, 'x') * map - INF * (1 - map) # (batch, enc_len, vocab + extra)
            # copy_score_max = copy_score_map.max(axis = 1) # (batch, vocab + extra)
            # copy_score = T.exp(copy_score_map - copy_score_max.dimshuffle(0, 'x', 1)).sum(axis = 1) # (batch, vocab + extra)
            # copy_score = copy_score_max + T.log(copy_score)
            # copy_log_probs = log_softmax(copy_score) # (batch, vocab + extra)
            # copy_mask = map.max(axis = 1) # (batch, vocab + extra)
            #
            # copy_weight_sigmoid, copy_weight_log_sigmoid, copy_weight_log_neg_sigmoid = get_sigmoid_comb(T.dot(hid, self.W_mode))
            # copy_weight_sigmoid = copy_weight_sigmoid.dimshuffle(0, 'x')
            # copy_weight_log_sigmoid = copy_weight_log_sigmoid.dimshuffle(0, 'x')
            # copy_weight_log_neg_sigmoid = copy_weight_log_neg_sigmoid.dimshuffle(0, 'x')
            #
            # log_probs_max = T.maximum(copy_log_probs[:, : self.word_cnt], gen_log_probs) # (batch, vocab)
            # vocab_log_probs = T.log(T.exp(copy_log_probs[:, : self.word_cnt] - log_probs_max) * copy_weight_sigmoid + T.exp(gen_log_probs - log_probs_max) * (1 - copy_weight_sigmoid)) + log_probs_max
            # vocab_copy_mask = copy_mask[:, : self.word_cnt]
            # vocab_log_probs = vocab_copy_mask * vocab_log_probs + (1 - vocab_copy_mask) * (copy_weight_log_neg_sigmoid + gen_log_probs)
            #
            # extra_copy_mask = copy_mask[:, self.word_cnt :]
            # extra_log_probs = copy_log_probs[:, self.word_cnt :] + copy_weight_log_sigmoid
            # extra_log_probs = extra_copy_mask * extra_log_probs + (1 - extra_copy_mask) * (- INF)
            #
            # combined_probs = T.zeros_like(copy_log_probs)
            # combined_probs = T.set_subtensor(combined_probs[:, : self.word_cnt], vocab_log_probs)
            # combined_probs = T.set_subtensor(combined_probs[:, self.word_cnt :], extra_log_probs)
            # prob = combined_probs
            #
            # next_input = T.cast(T.argmax(prob, axis = 1), 'int32')
            #
            # output_n = T.extra_ops.to_one_hot(next_input, self.word_cnt + self.extra_word_cnt) # (batch, vocab + extra)
            # output_n = T.batched_dot(output_n, map.dimshuffle(0, 2, 1)) # (batch, enc_len)
            # copy_probs = copy_score_no_map * output_n
            # copy_probs = copy_probs / (T.sum(copy_probs, axis = 1, keepdims = True) + 1e-8) # (batch, enc_len)
            # copy_hid = T.batched_dot(copy_probs, enc_feat) # (batch, units)

            return hid #[next_input, hid, copy_hid]

        step_fun = step

        if not isinstance(self.hid_init, lasagne.layers.Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)

        copy_hid_init = T.zeros((num_batch, self.num_units))
        input_init = T.cast(T.zeros((num_batch, )).fill(self.start_index), 'int32')

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        non_seqs += [enc_feat, enc_mask, self.W_emb]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            [token_out, hid_out, _] = unroll_scan(
                fn=step_fun,
                outputs_info=[input_init, hid_init, copy_hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=self.gen_len)[0]
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            [token_out, hid_out, _] = theano.scan(
                fn=step_fun,
                go_backwards=self.backwards,
                outputs_info=[input_init, hid_init, copy_hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True,
                n_steps=self.gen_len)[0]

        return token_out.dimshuffle(1, 0)

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, output, mask]
            step_fun = step_masked
        else:
            sequences = [input, output]
            step_fun = step

        if not isinstance(self.hid_init, lasagne.layers.Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)

        copy_hid_init = T.zeros((num_batch, self.num_units))

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        non_seqs += [enc_feat, enc_mask, self.W_emb]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            [hid_out, _, prob_out] = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init, copy_hid_init, None],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            [hid_out, _, prob_out] = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init, copy_hid_init, None],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            prob_out = prob_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            prob_out = prob_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                prob_out = prob_out[:, ::-1]

        return prob_out

class GRUCopyPureSampleLayer(lasagne.layers.MergeLayer):
    def __init__(self, num_units,
                 resetgate=lasagne.layers.Gate(W_cell=None),
                 updategate=lasagne.layers.Gate(W_cell=None),
                 hidden_update=lasagne.layers.Gate(W_cell=None,
                                    nonlinearity=lasagne.nonlinearities.tanh),
                 hid_init=lasagne.init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=False,
                 mask_input=None,
                 only_return_final=False,
                 l_enc_feat=None,
                 l_enc_mask=None,
                 l_map=None,
                 word_cnt=None,
                 extra_word_cnt=None,
                 W_emb=None,
                 W_gen=lasagne.init.GlorotUniform(),
                 W_copy=lasagne.init.GlorotUniform(),
                 W_mode=lasagne.init.Normal(),
                 unk_index=None,
                 start_index=None,
                 gen_len=None,
                 MRG_stream=None,
                 **kwargs):

        incomings = []
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if isinstance(hid_init, lasagne.layers.Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1

        incomings.append(l_enc_feat)
        self.enc_feat_index = len(incomings) - 1
        incomings.append(l_enc_mask)
        self.enc_mask_index = len(incomings) - 1
        incomings.append(l_map)
        self.map_index = len(incomings) - 1

        self.MRG_stream = MRG_stream

        # Initialize parent layer
        super(GRUCopyPureSampleLayer, self).__init__(incomings, **kwargs)

        self.learn_init = learn_init
        self.num_units = num_units
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final
        self.word_cnt = word_cnt
        self.extra_word_cnt = extra_word_cnt
        self.W_emb = W_emb
        self.unk_index = unk_index
        self.start_index = start_index
        self.gen_len = gen_len

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Input dimensionality is the output dimensionality of the input layer
        # num_inputs = np.prod(input_shape[2:])
        num_inputs = self.W_emb.get_value().shape[1]

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs + num_units + num_units, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in all parameters from gates
        (self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')

        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
         self.b_hidden_update, self.nonlinearity_hid) = add_gate_params(
             hidden_update, 'hidden_update')

        self.updategate = lasagne.layers.Gate(W_in = self.W_in_to_updategate, W_hid = self.W_hid_to_updategate, W_cell = None, b = self.b_updategate, nonlinearity = self.nonlinearity_updategate)
        self.resetgate = lasagne.layers.Gate(W_in = self.W_in_to_resetgate, W_hid = self.W_hid_to_resetgate, W_cell = None, b = self.b_resetgate, nonlinearity = self.nonlinearity_resetgate)
        self.hidden_update = lasagne.layers.Gate(W_in = self.W_in_to_hidden_update, W_hid = self.W_hid_to_hidden_update, W_cell = None, b = self.b_hidden_update, nonlinearity = self.nonlinearity_hid)

        # Initialize hidden state
        if isinstance(hid_init, lasagne.layers.Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

        self.W_gen = self.add_param(W_gen, (self.num_units, self.word_cnt), name = "W_gen")
        self.W_copy = self.add_param(W_copy, (self.num_units, self.num_units), name = "W_copy")
        self.W_mode = self.add_param(W_mode, (self.num_units, ), name = "W_mode")

    def get_output_shape_for(self, input_shapes):
        return input_shapes[self.enc_feat_index][0], self.gen_len

    def get_output_for(self, inputs, **kwargs):
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        if self.mask_incoming_index >= 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index >= 0:
            hid_init = inputs[self.hid_init_incoming_index]

        enc_feat = inputs[self.enc_feat_index]
        enc_mask = inputs[self.enc_mask_index]
        map = inputs[self.map_index] # (batch, enc_len, vocab + extra)

        num_batch = enc_feat.shape[0]

        # Stack input weight matrices into a (num_inputs, 3*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate, self.W_in_to_updategate,
             self.W_in_to_hidden_update], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1)

        # Stack gate biases into a (3*num_units) vector
        b_stacked = T.concatenate(
            [self.b_resetgate, self.b_updategate,
             self.b_hidden_update], axis=0)

        # When theano.scan calls step, input_n will be (n_batch, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        W_emb_comp = T.dot(T.ones((self.extra_word_cnt, 1)), self.W_emb[self.unk_index].dimshuffle('x', 0)) # (extra, emb)
        W_emb = T.concatenate([self.W_emb, W_emb_comp], axis = 0)

        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        def step(input_n, hid_previous, copy_hid_previous, *args):
            """
            input_n: (batch, ); each entry is the index; with extra vocabulary.
            hid_previous: (batch, units).
            prob_previous: (batch, vocab + extra).
            copy_hid_previous: (batch, units); a vector that combines the hidden states using the copy probabilities at the last time step.
            """
            input_emb = W_emb[input_n]

            # enc_feat: (batch, enc_len, units), hid_previous: (batch, units)
            att = T.batched_dot(enc_feat, hid_previous) # (batch, enc_len)
            att = T.nnet.softmax(att) * enc_mask # (batch, enc_len)
            att = att / (T.sum(att, axis = 1, keepdims = True) + 1e-8) # (batch, enc_len)
            att = T.batched_dot(att, enc_feat) # (batch, units)
            input_n = T.concatenate([input_emb, copy_hid_previous, att], axis = 1)

            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
            hid_input = T.dot(hid_previous, W_hid_stacked)

            if self.grad_clipping:
                input_n = theano.gradient.grad_clip(
                    input_n, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)

            if not self.precompute_input:
                # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            # Reset and update gates
            resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)

            # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
            hidden_update_in = slice_w(input_n, 2)
            hidden_update_hid = slice_w(hid_input, 2)
            hidden_update = hidden_update_in + resetgate*hidden_update_hid
            if self.grad_clipping:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
            hidden_update = self.nonlinearity_hid(hidden_update)

            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid = (1 - updategate)*hid_previous + updategate*hidden_update # (batch, units)

            gen_score = T.dot(hid, self.W_gen)
            gen_log_probs = log_softmax(gen_score)
            copy_score = T.batched_dot(T.tanh(T.dot(enc_feat, self.W_copy)), hid)
            copy_score -= copy_score.max(axis = 1, keepdims = True)
            copy_score_no_map = T.exp(copy_score)
            copy_score_map = copy_score.dimshuffle(0, 1, 'x') * map - INF * (1 - map) # (batch, enc_len, vocab + extra)
            copy_score_max = copy_score_map.max(axis = 1) # (batch, vocab + extra)
            copy_score = T.exp(copy_score_map - copy_score_max.dimshuffle(0, 'x', 1)).sum(axis = 1) # (batch, vocab + extra)
            copy_score = copy_score_max + T.log(copy_score)
            copy_log_probs = log_softmax(copy_score) # (batch, vocab + extra)
            copy_mask = map.max(axis = 1) # (batch, vocab + extra)

            copy_weight_sigmoid, copy_weight_log_sigmoid, copy_weight_log_neg_sigmoid = get_sigmoid_comb(T.dot(hid, self.W_mode))
            copy_weight_sigmoid = copy_weight_sigmoid.dimshuffle(0, 'x')
            copy_weight_log_sigmoid = copy_weight_log_sigmoid.dimshuffle(0, 'x')
            copy_weight_log_neg_sigmoid = copy_weight_log_neg_sigmoid.dimshuffle(0, 'x')

            log_probs_max = T.maximum(copy_log_probs[:, : self.word_cnt], gen_log_probs) # (batch, vocab)
            vocab_log_probs = T.log(T.exp(copy_log_probs[:, : self.word_cnt] - log_probs_max) * copy_weight_sigmoid + T.exp(gen_log_probs - log_probs_max) * (1 - copy_weight_sigmoid)) + log_probs_max
            vocab_copy_mask = copy_mask[:, : self.word_cnt]
            vocab_log_probs = vocab_copy_mask * vocab_log_probs + (1 - vocab_copy_mask) * (copy_weight_log_neg_sigmoid + gen_log_probs)

            extra_copy_mask = copy_mask[:, self.word_cnt :]
            extra_log_probs = copy_log_probs[:, self.word_cnt :] + copy_weight_log_sigmoid
            extra_log_probs = extra_copy_mask * extra_log_probs + (1 - extra_copy_mask) * (- INF)

            combined_probs = T.zeros_like(copy_log_probs)
            combined_probs = T.set_subtensor(combined_probs[:, : self.word_cnt], vocab_log_probs)
            combined_probs = T.set_subtensor(combined_probs[:, self.word_cnt :], extra_log_probs)
            # prob = combined_probs

            prob = T.exp(combined_probs)

            next_input = T.cast(T.argmax(self.MRG_stream.multinomial(pvals = prob), axis = 1), 'int32')

            output_n = T.extra_ops.to_one_hot(next_input, self.word_cnt + self.extra_word_cnt) # (batch, vocab + extra)
            output_n = T.batched_dot(output_n, map.dimshuffle(0, 2, 1)) # (batch, enc_len)
            copy_probs = copy_score_no_map * output_n
            copy_probs = copy_probs / (T.sum(copy_probs, axis = 1, keepdims = True) + 1e-8) # (batch, enc_len)
            copy_hid = T.batched_dot(copy_probs, enc_feat) # (batch, units)

            return [next_input, hid, copy_hid]

        step_fun = step

        if not isinstance(self.hid_init, lasagne.layers.Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)

        copy_hid_init = T.zeros((num_batch, self.num_units))
        input_init = T.cast(T.zeros((num_batch, )).fill(self.start_index), 'int32')

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        non_seqs += [enc_feat, enc_mask, self.W_gen, self.W_copy, self.W_mode, self.W_emb, map]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            [token_out, hid_out, _], self.updates = unroll_scan(
                fn=step_fun,
                outputs_info=[input_init, hid_init, copy_hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=self.gen_len) # [0]
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            [token_out, hid_out, _], self.updates = theano.scan(
                fn=step_fun,
                go_backwards=self.backwards,
                outputs_info=[input_init, hid_init, copy_hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True,
                n_steps=self.gen_len) # [0]

        return token_out.dimshuffle(1, 0)
