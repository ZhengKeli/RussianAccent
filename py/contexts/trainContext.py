from contexts.graphContext import *

from tools.dataTools import *

# load data
all_words_data, all_accents_data = load_cooked_data(conf.cooked_data_file)
train_words_data, test_words_data = split_data(all_words_data, [conf.train_proportion, conf.test_proportion])
train_accents_data, test_accents_data = split_data(all_accents_data, [conf.train_proportion, conf.test_proportion])

# setup summary
summary_writer = tf.summary.FileWriter(conf.log_dir)
summary_loss = tf.summary.scalar("summary_loss", sess.graph.get_tensor_by_name("train/loss_mean:0"))


# methods
def train_graph(train_count=1000, batch_size=10):
	return graphTools.train_graph_with_summary(
		sess,
		train_words_data, train_accents_data,
		summary_writer, summary_loss,
		train_count, batch_size
	)


def test_graph(test_count=200):
	return graphTools.test_graph(sess, test_words_data, test_accents_data, test_count)
