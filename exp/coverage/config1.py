
class config:
    # pkl_file = '../data/baixing.pairs.pkl'
    train_pkl = 'data/sutter_encounter.train.pkl'
    dev_pkl = 'data/sutter_encounter.dev.pkl'
    source_vocab_pkl = 'data/sutter_diag_vocab.pkl'
    target_vocab_pkl = 'data/sutter_drug_vocab_3.pkl'
    model_seed = 13
    vocab_size = 4000
    batch_size = 512
    source_len = 20
    target_len = 20
    char_embedding_size = 100
    enc_units = 256
    dec_units = 256
    grad_clipping = 10
    learning_rate = 1e-3
    reinforce_learning_rate = 1e-5
    max_loss_batch = 1
    print_loss_per = 20
    print_reinforce_per = 10
    max_epoch = 1000000

    dir = '../models/'
    saved_model_file = dir + 'rf_seed{}_{}d_lr{}_h{}.model'.format(model_seed, char_embedding_size, learning_rate, enc_units)

def get_config():
    return config()