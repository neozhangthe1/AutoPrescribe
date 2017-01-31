
class config:
    # pkl_file = '../data/baixing.pairs.pkl'
    train_pkl = 'data/mimic_encounter_gpi.train.pkl'
    dev_pkl = 'data/mimic_encounter_gpi.dev.pkl'
    source_vocab_pkl = 'data/mimic_diag_vocab.pkl'
    target_vocab_pkl = 'data/mimic_drug_vocab.pkl'
    model_seed = 13
    vocab_size = 4000
    batch_size = 512
    source_len = 50
    target_len = 50
    embedding_size = 100
    enc_units = 256
    dec_units = 256
    grad_clipping = 10
    learning_rate = 1e-3
    reinforce_learning_rate = 1e-5
    max_loss_batch = 1
    print_loss_per = 5
    print_reinforce_per = 10
    max_epoch = 1000000

    dir = 'build/'
    saved_model_file = dir + 'mimic_sort_seq2seq_len_{}_seed{}_{}d_lr{}_h{}.model'.format(source_len, model_seed, embedding_size, learning_rate, enc_units)

def get_config():
    return config()