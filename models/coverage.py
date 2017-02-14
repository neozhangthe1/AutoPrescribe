import io
import pickle

import lasagne
import numpy as np
import theano
from theano import tensor as T
from theano.compile.nanguardmode import NanGuardMode
from utils.data import dump

from models import layers
from utils import eval as eval_utils


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
        # T.sum(l_source.W)
        # l_s_gru_fw = lasagne.layers.GRULayer(l_source, config.enc_units, mask_input=l_source_mask_inputs,
        #                                      grad_clipping=config.grad_clipping)
        # l_s_gru_bw = lasagne.layers.GRULayer(l_source, config.enc_units, mask_input=l_source_mask_inputs,
        #                                      grad_clipping=config.grad_clipping)
        # l_source = lasagne.layers.ConcatLayer([l_s_gru_fw, l_s_gru_bw], axis=2)
        # l_source = lasagne.layers.GRULayer(l_source, config.enc_units, mask_input=l_source_mask_inputs,
        #                                    grad_clipping=config.grad_clipping)
        # l_source_last = lasagne.layers.ElemwiseSumLayer(l_source) #lasagne.layers.SliceLayer(l_source, -1, axis=1)

        l_target_outputs = layers.GRUCoverageTrainLayer(l_target_inputs, config.dec_units, mask_input=l_target_mask_inputs,
                                                    grad_clipping=config.grad_clipping, source_token_cnt=processor.source_vocab_size,
                                                    target_token_cnt=processor.target_vocab_size, l_enc_feat=l_source,
                                                    l_enc_mask=l_source_mask_inputs,
                                                    l_output=l_output, W_emb=self.W2,
                                                    unk_index=processor.get_char_index('UNK', False))#, hid_init=l_source_last)
        l_t = l_target_outputs
        l_target_outputs = lasagne.layers.ReshapeLayer(l_target_outputs, (-1, [2]))  # (batch * dec_len, vocab + extra)

        l_gen = layers.GRUCoverageTestLayer(config.dec_units, grad_clipping=config.grad_clipping,
                                        source_token_cnt=processor.source_vocab_size, target_token_cnt=processor.target_vocab_size,
                                        l_enc_feat=l_source, l_enc_mask=l_source_mask_inputs,
                                        W_emb=self.W2, resetgate=l_t.resetgate, updategate=l_t.updategate,
                                        hidden_update=l_t.hidden_update, #hid_init=l_source_last,
                                        unk_index=processor.get_char_index('UNK', False),
                                        start_index=processor.get_char_index('START', False), W_gen = l_t.W_gen, gen_len=config.target_len)
        l_att = layers.GRUCoverageAttLayer(config.dec_units, grad_clipping=config.grad_clipping,
                                        source_token_cnt=processor.source_vocab_size, target_token_cnt=processor.target_vocab_size,
                                        l_enc_feat=l_source, l_enc_mask=l_source_mask_inputs,
                                        W_emb=self.W2, resetgate=l_t.resetgate, updategate=l_t.updategate,
                                        hidden_update=l_t.hidden_update, #hid_init=l_source_last,
                                        unk_index=processor.get_char_index('UNK', False),
                                        start_index=processor.get_char_index('START', False), W_gen = l_t.W_gen, gen_len=config.target_len)
        self.l = l_target_outputs

        py = lasagne.layers.get_output(l_target_outputs)
        loss = (
        py * T.extra_ops.to_one_hot(target_outputs.flatten(), processor.target_vocab_size)).sum(
            axis=1)  # (batch * dec_len)
        loss = - (loss * target_mask_inputs.flatten()).mean()

        params = lasagne.layers.get_all_params(self.l, trainable=True)
        updates = lasagne.updates.adam(loss, params, learning_rate=config.learning_rate)

        gen_y = lasagne.layers.get_output(l_gen)

        gen_att = lasagne.layers.get_output(l_att)

        self.train_fn = theano.function(
                [source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs],
                None, updates=updates, on_unused_input='ignore',
                mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
        self.loss_fn = theano.function(
                [source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs],
                loss, on_unused_input='ignore',
                mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
        self.test_fn = theano.function([source_inputs, source_mask_inputs], gen_y, on_unused_input='ignore')
        self.att_fn = theano.function([source_inputs, source_mask_inputs], gen_att, on_unused_input='ignore')

        l_samp = layers.GRUCopyPureSampleLayer(config.dec_units, grad_clipping=config.grad_clipping,
                                               source_token_cnt=processor.source_vocab_size,
                                               target_token_cnt=processor.target_vocab_size,
                                               l_enc_feat=l_source, l_enc_mask=l_source_mask_inputs,
                                               W_emb=self.W2, resetgate=l_t.resetgate, updategate=l_t.updategate,
                                               hidden_update=l_t.hidden_update, #hid_init=l_source_last,
                                               unk_index=processor.get_char_index('UNK', False),
                                               start_index=processor.get_char_index('START', False),
                                               gen_len=config.target_len,
                                               W_gen=l_t.W_gen,
                                               MRG_stream=self.MRG_stream)  # (batch, dec_len)
        samp_y = lasagne.layers.get_output(l_samp)
        self.sample_fn = theano.function([source_inputs, source_mask_inputs], samp_y,
                                         updates=l_samp.updates, on_unused_input='ignore')

        reward_inputs = T.matrix()  # (batch, dec_len)
        reinforce_loss = (
        py * T.extra_ops.to_one_hot(target_outputs.flatten(), processor.target_vocab_size)).sum(
            axis=1)  # (batch * dec_len)
        reinforce_loss = - (reinforce_loss * target_mask_inputs.flatten() * reward_inputs.flatten()).mean()
        reinforce_updates = lasagne.updates.adam(reinforce_loss, params, learning_rate=config.reinforce_learning_rate)
        self.reinforce_fn = theano.function(
                [source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs,
                 reward_inputs], None, updates=reinforce_updates, on_unused_input='ignore')

        print('params', lasagne.layers.count_params(self.l, trainable=True))

    def comp_loss(self, data):
        p = self.processor
        config = self.config
        loss, cnt = 0, 0
        for step, (source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs,
                   refs) in enumerate(p.gen_batch(data, shuffle=False)):

            cur_loss = self.loss_fn(source_inputs, target_inputs, target_outputs, source_mask_inputs,
                                    target_mask_inputs)
            loss += cur_loss
            cnt += 1

            if step >= config.max_loss_batch - 1: break
        return loss / cnt

    def comp_reinforce_loss(self, data, scorer, max_batch=1):
        p, config = self.processor, self.config
        rewards, cnt = 0, 0
        for step, (source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs,
                   refs) in enumerate(p.gen_batch(data, shuffle=False)):
            predictions = []
            samp_y = self.test_fn(source_inputs, source_mask_inputs)
            for j in range(len(refs)):
                # refs[j].target_text = p.decode(samp_y[j], refs[j])
                predictions.append(p.decode(samp_y[j], refs[j]))
            source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs = p.gen_one_batch(
                refs)

            # instances = [[ref.target_text, ref.source_text] for ref in refs]
            instances = [[ref.target_text, predictions[i]] for i, ref in enumerate(refs)]
            rewards += np.array(scorer.predict(instances), dtype=np.float32).mean()
            cnt += 1

            if step >= max_batch - 1: break
        return rewards / cnt

    def save_params(self, filename):
        params = lasagne.layers.get_all_param_values(self.l)
        fout = open(filename, 'wb')
        pickle.dump(params, fout, pickle.HIGHEST_PROTOCOL)
        fout.close()

    def load_params(self, filename):
        fin = open(filename, "rb")
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
            # gen_att = self.att_fn(source_inputs, source_mask_inputs)
            # print(gen_y.shape)
            # print(type(gen_y))
            # print(gen_att.shape)
            # print(type(gen_att))
            for i in range(gen_y.shape[0]):
                if i >= 1 and training: break
                s = []
                for j in range(gen_y.shape[1]):
                    char_index = gen_y[i, j]
                    s.append(p.idx_to_target_token[char_index])

                if training:
                    print("S:", refs[i].source_text)
                    print("T:", refs[i].target_text)
                    print("Gen:", s)
                else:
                    if step % 1000 == 0 and i == 1:
                        print("step", step)
                        print("eval S:", refs[i].source_text)
                        print("eval T:", refs[i].target_text)
                        print("eval Gen:", s)
                    fout.write(u"S: {}\n".format(u" ".join(refs[i].source_text)))
                    fout.write(u"T: {}\n".format(u" ".join(refs[i].target_text))) #refs[i].target_text))
                    fout.write(u"Gen: {}\n".format(u" ".join(s)))
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
            gen_att = self.att_fn(source_inputs, source_mask_inputs, map_inputs)

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
                        self.save_params(config.saved_model_file + "_%s_%s" % (epoch, int(step / 1000) * 1000))
                    print('epoch', epoch, 'step', step)
                    print('train', train_loss, 'dev', dev_loss, 'min', min_measure)
                    self.do_eval()
                if step % 500 == 0:
                    self.do_eval(training = False, filename = '%s_%s_%s_seq2seq_e%s_s%s.txt' % (config.name, config.level, config.order, epoch, step), max_batch = 10000)
                    cnt = 0
                    results = []
                    input = []
                    truth = []
                    for line in open('%s_%s_%s_seq2seq_e%s_s%s.txt' % (config.name, config.level, config.order, epoch, step)):
                        if cnt % 3 == 0:
                            input = list(set(line.strip().split("S: ")[1].split(" ")))
                        if cnt % 3 == 1:
                            if len(line.strip().split("T: ")) <= 1:
                                truth = []
                                continue
                            truth = list(set(line.strip().split("T: ")[1].split(" ")))
                        if cnt % 3 == 2:
                            result = set(line.strip().split("Gen: ")[1].split("END")[0].strip().split(" "))
                            if '' in result:
                                result.remove('')
                            if len(truth) > 0:
                                results.append((input, truth, result))
                        cnt += 1
                    input_list, truth_list, prediction_list = eval_utils.get_results(results)
                    jaccard = eval_utils.get_average_jaccard(truth_list, prediction_list)
                    acc = eval_utils.get_average_accuracy(truth_list, prediction_list)
                    print("jaccard", jaccard, "acc", acc)
                    dump(results, "%s_%s_%s_result_seq2seq_e%s_s%s_jacc%s_acc%s.pkl" % (config.name, config.level, config.order, epoch, step, round(jaccard, 5), round(acc, 5)))

    def do_reinforce(self, scorer):
        p, config = self.processor, self.config

        max_reward = -1e6

        with open("reinforce_reward_%s_%s_per_1.txt" % (config.name, config.level), "w") as f_out:
            for epoch in range(self.config.max_epoch):
                for step, (source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs,
                           refs) in enumerate(p.gen_batch(p.train_data)):
                    samp_y = self.sample_fn(source_inputs, source_mask_inputs)
                    # print("samp_y", samp_y)
                    # gen_y = self.test_fn(source_inputs, source_mask_inputs)
                    # print("gen_y", gen_y)
                    predictions = []
                    for j in range(len(refs)):
                        predictions.append(p.decode(samp_y[j], refs[j]))
                        # if j == 0:
                        #     print(predictions)
                    source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs = p.gen_one_batch(refs)
                    # print(predictions[0])
                    # print(refs[0].target_text)
                    instances = [[ref.target_text, predictions[i]] for i, ref in enumerate(refs)]
                    rewards = np.array(scorer.predict(instances), dtype=np.float32)
                    rewards = np.tile(rewards, (config.target_len, 1)).transpose()  # (batch, dec_len)
                    # print(rewards)

                    self.reinforce_fn(source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs, rewards)

                    if step % config.print_reinforce_per == 0:
                        train_reward = self.comp_reinforce_loss(p.train_data, scorer, 1)
                        if step % 1 == 0:
                            dev_reward = self.comp_reinforce_loss(p.dev_data, scorer, 1)
                            print("full dev loss", dev_reward)
                            f_out.write("%s\t%s\t%s\n" % (epoch, step, dev_reward))
                            f_out.flush()
                        else:
                            dev_reward = self.comp_reinforce_loss(p.dev_data, scorer, 1)
                        if dev_reward > max_reward:
                            max_reward = dev_reward
                        #     self.save_params(config.saved_model_file)
                        print('epoch', epoch, 'step', step)
                        print('train', train_reward, 'dev', dev_reward, 'max', max_reward)
