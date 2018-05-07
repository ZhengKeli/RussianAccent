import numpy as np

import conf


def cook_data_words(words):
	words_data = list(filter(lambda e: e is not None, [cook_data_word(word) for word in words]))
	words_data = np.array(words_data, dtype=np.int32)
	return words_data


def cook_data_word(word):
	if len(word) > conf.max_word_length:
		return None
	
	word_data = np.zeros([conf.max_word_length], np.int32)
	for i in range(conf.max_word_length):
		if i >= len(word):
			break
		
		letter_data = conf.letters.find(word[i])
		if letter_data == -1:
			return None
		
		word_data[i] = letter_data
	return word_data


def load_cooked_data(cooked_data_file):
	all_data = np.load(cooked_data_file)
	all_data_words = all_data["words"]
	all_data_accents = all_data["accents"]
	return all_data_words, all_data_accents


def split_data(all_data, proportions):
	current_index = 0
	pieces_data = []
	for proportion in proportions:
		length = int(len(all_data) * proportion)
		head_index = current_index
		tail_index = head_index + length
		pieces_data.append(all_data[head_index: tail_index])
		current_index = tail_index
	return pieces_data
