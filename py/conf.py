import os

# file conf
version_name = "v2"
root_dir = "..\\"

train_data_dir = os.path.join(root_dir, "trainData")
standard_data_file = os.path.join(train_data_dir, "standardData.txt")
cooked_data_file = os.path.join(train_data_dir, "cookedData.npz")

log_dir = os.path.join(root_dir, "log", version_name)

graph_dir = os.path.join(root_dir, "graph", version_name)
graph_file = os.path.join(graph_dir, version_name)
graph_meta_file = os.path.join(graph_dir, version_name + ".meta")

# graph conf
max_word_length = 32

# data conf
letters = " абвгдеёжзийклмнопрстуфхцчшщъыьэюя"

# train conf
train_proportion = 0.8
test_proportion = 1 - train_proportion
