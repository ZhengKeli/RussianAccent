import os

# graph conf
word_length = 32
letter_count = 33 + 1
train_proportion = 0.8
train_batch_size = 10
test_proportion = 1 - train_proportion
letters = " абвгдеёжзийклмнопрстуфхцчшщъыьэюя"

# file conf
version_name = "v2"
root_dir = ".\\"

trainData_dir = os.path.join(root_dir, "trainData")
standardData_file = os.path.join(trainData_dir, "standardData.txt")
cookedData_file = os.path.join(trainData_dir, "cookedData.npz")

log_dir = os.path.join(root_dir, "log", version_name)

graph_dir = os.path.join(root_dir, "graph", version_name)
graph_file = os.path.join(graph_dir, version_name)
graph_meta_file = os.path.join(graph_dir, version_name + ".meta")
