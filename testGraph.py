import numpy as np
import tensorflow as tf

import conf

# load data
all_data = np.load(conf.cookedData_file)
all_data_words = all_data["words"]
all_data_accents = all_data["accents"]

all_data_count = len(all_data_words)
train_data_count = int(all_data_count * conf.train_proportion)
test_data_count = all_data_count - train_data_count

test_data_words = all_data_words[train_data_count:all_data_count]
test_data_accents = all_data_accents[train_data_count:all_data_count]

# load graph
sess = tf.Session()
tf.train.import_meta_graph(conf.graph_meta_file) \
	.restore(sess, tf.train.latest_checkpoint(conf.graph_dir))
print("graph loaded")

# test
test_batch_size = 200
batch_index = np.random.randint(test_data_count, size=[test_batch_size])
val_word = test_data_words.take(batch_index, axis=0)
val_accent = test_data_accents.take(batch_index, axis=0)
val_correct_rate, val_training_step = sess.run(["train/correct_rate:0", "train/training_step:0"], {
	"input/word:0": val_word,
	"input/accent:0": val_accent,
})
print("[", val_training_step, "] test: correct_rate=", val_correct_rate * 100, "%")
