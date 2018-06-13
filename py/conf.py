import os

# file conf
root_dir = "..\\"

train_data_dir = os.path.join(root_dir, "trainData")
standard_data_file = os.path.join(train_data_dir, "standardData.txt")
cooked_data_file = os.path.join(train_data_dir, "cookedData.npz")

logs_dir = os.path.join(root_dir, "logs")
graphs_dir = os.path.join(root_dir, "graphs")

# graph conf
max_word_length = 32
default_version = "v3"

# data conf
letters = " абвгдеёжзийклмнопрстуфхцчшщъыьэюя"

# train conf
train_proportion = 0.8
test_proportion = 1 - train_proportion


# dynamic
def get_log_dir(version):
	return os.path.join(logs_dir, version)


def get_graph_dir(version, step):
	return os.path.join(graphs_dir, version, "step_" + str(step))


def get_graph_file(version, step):
	return os.path.join(get_graph_dir(version, step), "graph")


def get_graph_meta_file(version, step):
	return os.path.join(get_graph_dir(version, step), "graph.meta")


def get_latest_step(version):
	latest_step = None
	for dir_name in os.listdir(os.path.join(graphs_dir, version)):
		if not dir_name.startswith("step_"):
			continue
		step_string = dir_name[5:]
		if not step_string.isdigit():
			continue
		step = int(step_string)
		if latest_step is None:
			latest_step = step
			continue
		elif step > latest_step:
			latest_step = step
			continue
	return latest_step
