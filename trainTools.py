import numpy as np
import tensorflow as tf

import conf
import tools

# load data
all_data = np.load(conf.cookedData_file)
all_data_words = all_data["words"]
all_data_accents = all_data["accents"]

all_data_count = len(all_data_words)
train_data_count = int(all_data_count * conf.train_proportion)
test_data_count = all_data_count - train_data_count

train_data_words = all_data_words[0:train_data_count]
train_data_accents = all_data_accents[0:train_data_count]

test_data_words = all_data_words[train_data_count:all_data_count]
test_data_accents = all_data_accents[train_data_count:all_data_count]

# load graph
sess = tf.Session()
tf.train.import_meta_graph(conf.graph_meta_file) \
	.restore(sess, tf.train.latest_checkpoint(conf.graph_dir))
print("graph loaded")

# summary
summary_writer = tf.summary.FileWriter(conf.log_dir)
summary_loss = tf.summary.scalar("summary_loss", sess.graph.get_tensor_by_name("train/loss_mean:0"))


# processes
def process_train(train_count=1000):
	for i in range(train_count):
		batch_index = np.random.randint(train_data_count, size=[conf.train_batch_size])
		val_word = train_data_words.take(batch_index, axis=0)
		val_accent = train_data_accents.take(batch_index, axis=0)
		val_training, val_loss, val_training_step, val_summary_loss = sess.run(
			["train/training", "train/loss:0", "train/training_step:0", summary_loss],
			{"input/word:0": val_word, "input/accent:0": val_accent, "train/training_rate:0": 0.002}
		)
		print("[", val_training_step, "] train: loss=", np.mean(val_loss))
		summary_writer.add_summary(val_summary_loss, val_training_step)
	print("trained", train_count, "times")


def process_test(test_batch_size=200):
	batch_index = np.random.randint(test_data_count, size=[test_batch_size])
	val_word = test_data_words.take(batch_index, axis=0)
	val_accent = test_data_accents.take(batch_index, axis=0)
	val_correct_rate, val_training_step = sess.run(["train/correct_rate:0", "train/training_step:0"], {
		"input/word:0": val_word,
		"input/accent:0": val_accent,
	})
	print("[", val_training_step, "] test: correct_rate=", val_correct_rate * 100, "%")


def process_use(words):
	# cook data
	words_data = []
	for word in words:
		word_data = tools.parse_word(word)
		if word_data is not None:
			words_data.append(word_data)
	if len(words_data) == 0:
		print("no recognizable words!")
		return
	
	# run graph
	words_data = np.array(words_data, dtype=np.int32)
	val_predict_accent = sess.run("analyzer/predict_accent:0", {
		"input/word:0": words_data,
	})
	
	# print
	print("Prediction:")
	for i in range(len(words)):
		print_string = words[i]
		insert_index = val_predict_accent[i] + 1
		print_string = print_string[0:insert_index] + "'" + print_string[insert_index:len(print_string)]
		print(print_string)


def process_save():
	tf.train.Saver().save(sess, conf.graph_file)
	print("graph saved")


# commands
while True:
	command = input("next command:").split()
	if command[0] == "train":
		if len(command) == 1:
			process_train()
		elif len(command) == 2:
			process_train(int(command[1]))
	elif command[0] == "test":
		if len(command) == 1:
			process_test()
		elif len(command) == 2:
			process_test(int(command[1]))
	elif command[0] == "save":
		process_save()
	elif command[0] == "use":
		process_use(command[1:len(command)])
	elif command[0] == "exit":
		break
	print()
print("bye!")
