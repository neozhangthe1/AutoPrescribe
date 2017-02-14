
class config:
    # pkl_file = '../data/baixing.pairs.pkl'
    name = "sutter"
    level = 2
    order = "freq"
    train_pkl = 'data/sutter_encounters_%s.train_%s.pkl' % (level, order)
    dev_pkl = 'data/sutter_encounters_%s.test_%s.pkl' % (level, order)
    source_vocab_pkl = 'data/sutter_diag_vocab.pkl'
    target_vocab_pkl = 'data/sutter_drug_vocab_%s.pkl' % level
    model_seed = 13
    vocab_size = 4000
    batch_size = 256
    source_len = 20
    target_len = 20
    embedding_size = 200
    enc_units = 200
    dec_units = 200
    grad_clipping = 10
    learning_rate = 1e-3
    reinforce_learning_rate = 1e-5
    max_loss_batch = 1
    print_loss_per = 5
    print_reinforce_per = 10
    max_epoch = 1000000

    # dir = 'build/'
    # saved_model_file = dir + 'sutter_seq2seq_sorted_seed{}_{}d_lr{}_h{}.model'.format(model_seed, embedding_size, learning_rate, enc_units)

def get_config():
    return config()