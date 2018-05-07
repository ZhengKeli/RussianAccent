from tools.dataTools import *

all_words_data, all_accents_data = load_cooked_data(conf.cooked_data_file)
train_words_data, test_words_data = split_data(all_words_data, [conf.train_proportion, conf.test_proportion])
train_accents_data, test_accents_data = split_data(all_accents_data, [conf.train_proportion, conf.test_proportion])
