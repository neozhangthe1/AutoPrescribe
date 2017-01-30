import io
import pickle

import lasagne
import numpy as np
import theano
from theano import tensor as T
from theano.compile.nanguardmode import NanGuardMode

from models import layers


class CoverageModel:
    def __init__(self, processor, config):
        self.config = config
        self.processor = processor

        lasagne.random.set_rng(np.random)
        np.random.seed(config.model_seed)

        self.MRG_stream = theano.sandbox.rng_mrg.MRG_RandomStreams(seed=config.model_seed)

        self.build()

    def build(self):
        config = self.config
        processor = self.processor

        source_inputs = T.imatrix()
        target_inputs = T.imatrix()
        target_outputs = T.imatrix()
        source_mask_inputs = T.matrix()
        target_mask_inputs = T.matrix()
        # map_inputs = T.tensor3()

        l_source_inputs = lasagne.layers.InputLayer(shape=(None, config.source_len), input_var=source_inputs)
        l_target_inputs = lasagne.layers.InputLayer(shape=(None, config.target_len), input_var=target_inputs)
        l_output = lasagne.layers.InputLayer(shape=(None, config.target_len), input_var=target_outputs)
        l_source_mask_inputs = lasagne.layers.InputLayer(shape=(None, config.source_len), input_var=source_mask_inputs)
        l_target_mask_inputs = lasagne.layers.InputLayer(shape=(None, config.target_len), input_var=target_mask_inputs)
        # l_map_inputs = lasagne.layers.InputLayer(shape=(None, config.source_len, processor.target_vocab_size),
        #                                          input_var=map_inputs)

        l_source = lasagne.layers.EmbeddingLayer(l_source_inputs, processor.source_vocab_size, config.embedding_size)
        l_target = lasagne.layers.EmbeddingLayer(l_target_inputs, processor.target_vocab_size, config.embedding_size)
        self.W1 = l_source.W
        self.W2 = l_target.W
        l_s_gru_fw = lasagne.layers.GRULayer(l_source, config.enc_units, mask_input=l_source_mask_inputs,
                                             grad_clipping=config.grad_clipping)
        l_s_gru_bw = lasagne.layers.GRULayer(l_source, config.enc_units, mask_input=l_source_mask_inputs,
                                             grad_clipping=config.grad_clipping)
        l_source = lasagne.layers.ConcatLayer([l_s_gru_fw, l_s_gru_bw], axis=2)
        l_source = lasagne.layers.GRULayer(l_source, config.enc_units, mask_input=l_source_mask_inputs,
                                           grad_clipping=config.grad_clipping)
        l_source_last = lasagne.layers.SliceLayer(l_source, -1, axis=1)

        l_target_outputs = layers.GRUCoverageTrainLayer(l_target_inputs, config.dec_units, mask_input=l_target_mask_inputs,
                                                    grad_clipping=config.grad_clipping, source_token_cnt=processor.source_vocab_size,
                                                    target_token_cnt=processor.target_vocab_size, l_enc_feat=l_source,
                                                    l_enc_mask=l_source_mask_inputs,
                                                    l_output=l_output, W_emb=self.W2,
                                                    unk_index=processor.get_char_index('UNK', False), hid_init=l_source_last)
        l_t = l_target_outputs
        l_target_outputs = lasagne.layers.ReshapeLayer(l_target_outputs, (-1, [2]))  # (batch * dec_len, vocab + extra)

        l_gen = layers.GRUCoverageTrainLayer(l_target_inputs, config.dec_units, mask_input=l_target_mask_inputs,
                                                    grad_clipping=config.grad_clipping, source_token_cnt=processor.source_vocab_size,
                                                    target_token_cnt=processor.target_vocab_size, l_enc_feat=l_source,
                                                    l_enc_mask=l_source_mask_inputs,
                                                    l_output=l_output, W_emb=self.W2,
                                                    unk_index=processor.get_char_index('UNK', False), hid_init=l_source_last)
            # layers.GRUCoverageTrainLayer(config.dec_units, config.dec_units,  grad_clipping=config.grad_clipping,
            #                             source_token_cnt=processor.source_vocab_size, target_token_cnt=processor.target_vocab_size,
            #                             l_enc_feat=l_source, l_enc_mask=l_source_mask_inputs,
            #                             W_emb=self.W2, resetgate=l_t.resetgate, updategate=l_t.updategate,
            #                             hidden_update=l_t.hidden_update, hid_init=l_source_last,
            #                             unk_index=processor.get_char_index('UNK', False),
            #                             start_index=processor.get_char_index('START', False), gen_len=config.target_len)
        self.l = l_target_outputs

        py = lasagne.layers.get_output(l_target_outputs)
        loss = (
        py * T.extra_ops.to_one_hot(target_outputs.flatten(), processor.target_vocab_size)).sum(
            axis=1)  # (batch * dec_len)
        loss = - (loss * target_mask_inputs.flatten()).mean()

        params = lasagne.layers.get_all_params(self.l, trainable=True)
        updates = lasagne.updates.adam(loss, params, learning_rate=config.learning_rate)

        gen_y = lasagne.layers.get_output(l_gen)

        self.train_fn = theano.function(
                [source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs],
                None, updates=updates, on_unused_input='ignore',
                mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
        self.loss_fn = theano.function(
                [source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs],
                loss, on_unused_input='ignore',
                mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
        self.test_fn = theano.function([source_inputs, source_mask_inputs], gen_y, on_unused_input='ignore')

        # l_samp = layers.GRUCopyPureSampleLayer(config.dec_units, grad_clipping=config.grad_clipping,
        #                                        word_cnt=processor.char_cnt, extra_word_cnt=processor.extra_char_cnt,
        #                                        l_enc_feat=l_source, l_enc_mask=l_source_mask_inputs,
        #                                        W_emb=self.W2, resetgate=l_t.resetgate, updategate=l_t.updategate,
        #                                        hidden_update=l_t.hidden_update, hid_init=l_source_last,
        #                                        unk_index=processor.get_char_index('UNK'),
        #                                        start_index=processor.get_char_index('START'), gen_len=config.target_len,
        #                                        MRG_stream=self.MRG_stream)  # (batch, dec_len)
        # samp_y = lasagne.layers.get_output(l_samp)
        # self.sample_fn = theano.function([source_inputs, source_mask_inputs, map_inputs], samp_y,
        #                                  updates=l_samp.updates, on_unused_input='ignore')

        # reward_inputs = T.matrix()  # (batch, dec_len)
        # reinforce_loss = (
        # py * T.extra_ops.to_one_hot(target_outputs.flatten(), processor.char_cnt + processor.extra_char_cnt)).sum(
        #     axis=1)  # (batch * dec_len)
        # reinforce_loss = - (reinforce_loss * target_mask_inputs.flatten() * reward_inputs.flatten()).mean()
        # reinforce_updates = lasagne.updates.adam(reinforce_loss, params, learning_rate=config.reinforce_learning_rate)
        # self.reinforce_fn = theano.function(
        #         [source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs, map_inputs,
        #          reward_inputs], None, updates=reinforce_updates, on_unused_input='ignore')

        print('params', lasagne.layers.count_params(self.l, trainable=True))

    def comp_loss(self, data):
        p = self.processor
        config = self.config
        loss, cnt = 0, 0
        for step, (source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs, map_inputs,
                   refs) in enumerate(p.gen_batch(data, shuffle=False)):

            cur_loss = self.loss_fn(source_inputs, target_inputs, target_outputs, source_mask_inputs,
                                    target_mask_inputs, map_inputs)
            loss += cur_loss
            cnt += 1

            if step >= config.max_loss_batch - 1: break
        return loss / cnt

    def comp_reinforce_loss(self, data, scorer):
        p, config = self.processor, self.config
        rewards, cnt = 0, 0
        for step, (source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs, map_inputs,
                   refs) in enumerate(p.gen_batch(data, shuffle=False)):

            samp_y = self.test_fn(source_inputs, source_mask_inputs, map_inputs)
            for j in range(len(refs)):
                refs[j].target_text = p.decode(samp_y[j], refs[j])
            source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs, map_inputs = p.gen_one_batch(
                refs)

            instances = [[ref.target_text, ref.source_text] for ref in refs]
            rewards += np.array(scorer.predict(instances), dtype=np.float32).mean()
            cnt += 1

            if step >= config.max_loss_batch - 1: break
        return rewards / cnt

    def save_params(self, filename):
        params = lasagne.layers.get_all_param_values(self.l)
        fout = open(filename, 'w')
        pickle.dump(params, fout, pickle.HIGHEST_PROTOCOL)
        fout.close()

    def load_params(self, filename):
        fin = open(filename)
        params = pickle.load(fin)
        lasagne.layers.set_all_param_values(self.l, params)
        fin.close()

    def do_eval(self, training=True, filename=None, max_batch=None):
        p = self.processor
        if not training:
            fout = io.open(filename, 'w', encoding='utf-8')
        for step, (source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs,
                   refs) in enumerate(p.gen_batch(p.dev_data)):
            gen_y = self.test_fn(source_inputs, source_mask_inputs)
            for i in range(gen_y.shape[0]):
                if i >= 1 and training: break
                s = []
                for j in range(gen_y.shape[1]):
                    char_index = gen_y[i, j]
                    if char_index < p.char_cnt:
                        s.append(p.index2char[char_index])
                    else:
                        s.append(refs[i].extras[char_index - p.char_cnt])

                if training:
                    print("S:", refs[i].source_text)
                    print("T:", refs[i].target_text)
                    print("Gen:", u"".join(s))
                else:
                    fout.write(u"S: {}\n".format(refs[i].source_text))
                    fout.write(u"T: {}\n".format(refs[i].target_text))
                    fout.write(u"Gen: {}\n".format(u"".join(s)))
            if max_batch is not None and step >= max_batch - 1: break
            if training: break
        if not training:
            fout.close()

    def do_generate(self, data, training=False):
        """
        data: list of pairs. The second element of the pair should be some random placeholder, like empty strings.
        return: list of pairs. The first element is the source text, and the second element is the generated text.
        """
        p = self.processor
        ret = []
        for step, (source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs, map_inputs,
                   refs) in enumerate(p.gen_batch(data, shuffle=False)):
            gen_y = self.test_fn(source_inputs, source_mask_inputs, map_inputs)
            for i in range(gen_y.shape[0]):
                if i >= 1 and training: break
                s = []
                for j in range(gen_y.shape[1]):
                    char_index = gen_y[i, j]
                    if char_index < p.char_cnt:
                        s.append(p.index2char[char_index])
                    else:
                        s.append(refs[i].extras[char_index - p.char_cnt])

                ret.append([refs[i].source_text, u"".join(s)])
        return ret

    def do_train(self):
        p = self.processor
        config = self.config

        min_measure = 1e6
        print("start train")

        for epoch in range(self.config.max_epoch):
            for step, (source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs,
                       refs) in enumerate(p.gen_batch(p.train_data)):
                self.train_fn(source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs)
                if step % config.print_loss_per == 0:
                    train_loss = self.comp_loss(p.train_data)
                    dev_loss = self.comp_loss(p.dev_data)
                    if dev_loss < min_measure:
                        min_measure = dev_loss
                        self.save_params(config.saved_model_file)
                    print('epoch', epoch, 'step', step)
                    print('train', train_loss, 'dev', dev_loss, 'min', min_measure)
                    self.do_eval()

    def do_reinforce(self, scorer):
        p, config = self.processor, self.config

        max_reward = -1e6

        for epoch in range(self.config.max_epoch):
            for step, (source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs,
                       refs) in enumerate(p.gen_batch(p.train_data)):
                samp_y = self.sample_fn(source_inputs, source_mask_inputs)
                for j in range(len(refs)):
                    refs[j].target_text = p.decode(samp_y[j], refs[j])
                    if j == 0:
                        print(u"".join(refs[j].target_text))
                source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs = p.gen_one_batch(
                    refs)

                instances = [[ref.target_text, ref.source_text] for ref in refs]
                rewards = np.array(scorer.predict(instances), dtype=np.float32)
                rewards = np.tile(rewards, (config.target_len, 1)).transpose()  # (batch, dec_len)

                self.reinforce_fn(source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs, rewards)

                if step % config.print_reinforce_per == 0:
                    train_reward = self.comp_reinforce_loss(p.train_data, scorer)
                    dev_reward = self.comp_reinforce_loss(p.dev_data, scorer)
                    if dev_reward > max_reward:
                        max_reward = dev_reward
                        self.save_params(config.saved_model_file)
                    print('epoch', epoch, 'step', step)
                    print('train', train_reward, 'dev', dev_reward, 'max', max_reward)
