from contexts.graphContext import *
from tools.dataTools import cook_data_words

while True:
	# input words
	words = input("Type the word(s):\n")
	if words == "exit":
		break
	words = words.split()
	
	# cook data
	words_data = cook_data_words(words)
	
	# run graph
	val_predict_accent = use_graph(words_data)
	
	# print
	print("Prediction:")
	for i in range(len(words)):
		print_string = words[i]
		insert_index = val_predict_accent[i] + 1
		print_string = print_string[0:insert_index] + "'" + print_string[insert_index:len(print_string)]
		print(print_string)

print("bye!")
