
class config:
    # pkl_file = '../data/baixing.pairs.pkl'
    name = "mimic"
    level = 2
    order = "freq"
    train_pkl = 'data/mimic_encounters_%s.train_%s.pkl' % (level, order)
    dev_pkl = 'data/mimic_encounters_%s.test_%s.pkl' % (level, order)
    source_vocab_pkl = 'data/mimic_diag_vocab.pkl'
    target_vocab_pkl = 'data/mimic_drug_vocab_%s.pkl' % level
    model_seed = 13
    vocab_size = 4000
    batch_size = 256
    source_len = 50
    target_len = 50
    embedding_size = 200
    enc_units = 200
    dec_units = 200
    grad_clipping = 10
    learning_rate = 1e-3
    reinforce_learning_rate = 1e-6
    max_loss_batch = 1
    print_loss_per = 5
    print_reinforce_per = 10
    max_epoch = 1000000

    # dir = 'build/'
    # saved_model_file = dir + 'mimic_new_sorted_seq2seq_len_{}_seed{}_{}d_lr{}_h{}.model'.format(source_len, model_seed, embedding_size, learning_rate, enc_units)

def get_config():
    return config()