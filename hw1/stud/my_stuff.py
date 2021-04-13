import os
import time
import jsonlines
import numpy as np

#from tqdm import tqdm
from typing import List, Tuple, Any, Dict

"""
    Using the provided functions to load the data, as in evaluate.py:
        - count(l: List[Any]) -> Dict[Any, int];
        - read_dataset(path: str) -> Tuple[List[Dict], List[str]]; 
"""
def count(l: List[Any]) -> Dict[Any, int]:
    d = {}
    for e in l:
        d[e] = 1 + d.get(e, 0)
    return d

def read_dataset(path: str) -> Tuple[List[Dict], List[str]]:

    sentence_pairs = []
    labels = []

    with jsonlines.open(path) as f:
        for obj in f:
            labels.append(obj.pop('label'))

            obj["sentence1"] = [word.lower() for word in re.split("\W", obj["sentence1"].lower()) if word]
            obj["sentence2"] = [word.lower() for word in re.split("\W", obj["sentence2"].lower()) if word]
            sentence_pairs.append(obj)

    assert len(sentence_pairs) == len(labels)
    return sentence_pairs, labels

#####################################################################
import re
import collections
from torch.utils.data import IterableDataset

"""
    My classes & functions
"""
class Word2VecDataset(IterableDataset):
    # TODO
    def __init__(self, data_path: str, vocab_size: int, unk_token: str, sep_token: str, window_size: int):
        """
        Args:
            data_file   : Path to the dataset file.
            vocab_size  : Maximum amount of words that we want to embed.
            unk_token   : How will unknown words represented (e.g. 'UNK').
            window_size : Number of words to consider as context.
        """
        self.sep_token = sep_token
        # [[w1,s1, w2,s1, ..., w|s1|,s1], [w1,s2, w2,s2, ..., w|s2|,s2], ..., [w1,sn, ..., w|sn|,sn]]
        self.window_size = window_size 
        self.data_json = self.read_dataset(data_path)  # tuple(s_pairs,labels)
        self.build_vocabulary(vocab_size, unk_token, sep_token)

    # TODO
    def __iter__(self):
        """TODO
        Overwrites the __iter__() method of the superclass (torch.utils.data.IterableDataset).
        """
        sentence_pairs = self.data_json[0]
        for spair in sentence_pairs:
            sentence = self.merge_pair(spair=spair, sep_token=self.sep_token) #RABARBARO
            len_sentence = len(sentence)

            for input_idx in range(len_sentence):
                current_word = sentence[input_idx]
                # must be a word in the vocabulary
                if current_word in self.word2id and self.keep_word(current_word):
                    # left and right window indices
                    min_idx = max(0, input_idx - self.window_size)
                    max_idx = min(len_sentence, input_idx + self.window_size)

                    window_idxs = [x for x in range(min_idx, max_idx) if x != input_idx]
                    for target_idx in window_idxs:
                        # must be a word in the vocabulary
                        if sentence[target_idx] in self.word2id:
                            # index of target word in vocab
                            target = self.word2id[sentence[target_idx]]
                            # index of input word
                            current_word_id = self.word2id[current_word]
                            output_dict = {'targets':target, 'inputs':current_word_id}

                            yield output_dict

    def merge_pair(self, spair: dict, sep_token: str):
        """
        Merge the sentences of a pair into one context, where the two are separated bt the sep_token.
        # may include this in tokenize line #RABARBARO
        """
        s1 = spair["sentence1"]
        s2 = spair["sentence2"]
        
        s1.append(sep_token)
        s1.extend(s2)
        return s1

    def read_dataset(self, data_path: str):
        """
        Reads the dataset and converts each sentence in the input file into a list of tokenized words.
        """       
        sentence_pairs = []
        labels = []

        with jsonlines.open(data_path) as f:
            for obj in f:
                labels.append(obj.pop('label'))

                obj["sentence1"] = self.tokenize_line(obj["sentence1"])
                obj["sentence2"] = self.tokenize_line(obj["sentence2"])
                sentence_pairs.append(obj)

        assert len(sentence_pairs) == len(labels)
        return sentence_pairs, labels

    def tokenize_line(self, line: str, pattern='\W'):
        """
        Tokenizes a single line (e.g. "The pen is on the table" -> ["the, "pen", "is", "on", "the", "table"])
        """
        return [word.lower() for word in re.split(pattern, line.lower()) if word]

    # TODO
    def build_vocabulary(self, vocab_size: int, unk_token: str, sep_token: str):
        """ TODO
        Defines the vocabulary to be used. Builds a mapping (word, index) for
        each word in the vocabulary.

        Args:
            vocab_size (int): size of the vocabolary
            unk_token (str): token to associate with unknown words
        """
        counter_list = []
        # context is a list of tokens within a single sentence
        for spair in self.data_json[0]:
            context = self.merge_pair(spair=spair, sep_token=sep_token)
            counter_list.extend(context)
        counter = collections.Counter(counter_list)
        counter_len = len(counter)
        print("Number of distinct words: {}".format(counter_len))

        # consider only the (vocab size -1) most common words to build the vocab
        dictionary = {key: index for index, (key, _) in enumerate(counter.most_common(vocab_size - 1))}
        assert unk_token not in dictionary
        # all the other words are mapped to UNK
        dictionary[unk_token] = vocab_size - 1
        self.word2id = dictionary

        # dictionary with (word, frequency) pairs -- including only words that are in the vocabulary
        dict_counts = {x: counter[x] for x in dictionary if x is not unk_token}
        self.frequency = dict_counts
        self.tot_occurrences = sum(dict_counts[x] for x in dict_counts)

        print("Total occurrences of words in dictionary: {}".format(self.tot_occurrences))

        less_freq_word = min(dict_counts, key=counter.get)
        print("Less frequent word in dictionary appears {} times ({})".format(dict_counts[less_freq_word],
                                                                              less_freq_word))

        # index to word
        self.id2word = {value: key for key, value in dictionary.items()}

        # data is the text converted to indexes, as list of lists
        data = []
        # for each sentence
        for spair in self.data_json[0]:
            sentence = self.merge_pair(spair=spair, sep_token=sep_token)
            paragraph = []
            # for each word in the sentence
            for i in sentence:
                id_ = dictionary[i] if i in dictionary else dictionary[unk_token]
                if id_ == dictionary[unk_token]:
                    continue
                paragraph.append(id_)
            data.append(paragraph)
        # list of lists of indices, where each sentence is a list of indices, ignoring UNK
        self.data_idx = data

    # TODO
    def keep_word(self, word):
        '''TODO
        Implements negative sampling and returns true if we can keep the occurrence as training instance.
        '''
        z = self.frequency[word] / self.tot_occurrences
        p_keep = np.sqrt(z / 10e-3) + 1
        p_keep *= 10e-3 / z # higher for less frequent instances
        return np.random.rand() < p_keep # toss a coin and compare it to p_keep to keep the word

#####################################################################
UNK = "UNK"
SEP = "SEP"
VOCAB_SIZE = 10000

if __name__ == '__main__':
    print("################## my_stuff test code ################")
    #os.chdir("../../")

    print("[INFO]: Loading data ...")
    dev_data_path = "data/dev.jsonl"

    try:
        print(f"data path: '{dev_data_path}'") 
        sentence_pairs, labels = read_dataset(dev_data_path)

    except FileNotFoundError as e:
        print(f'Evaluation crashed because {dev_data_path} does not exist')
        exit(1)

    except Exception as e:
        print(f'Evaluation crashed. Most likely, the file you gave is not in the correct format')
        print(f'Printing error found')
        print(e, exc_info=True)
        exit(1)
    
    print("labels:        ",len(labels))
    print("sentence pairs:",len(sentence_pairs))
    print("[INFO]: data loaded successfully.")

    """
    for spair in sentence_pairs:
        print(f"[INFO]: showing sentence pair -> { spair['id'] } ...")
        print("target:",spair["lemma"])
        print("s1:",spair["sentence1"])
        print("s2:",spair["sentence2"])

        s1 = spair["sentence1"]
        s2 = spair["sentence2"]
        
        s1.append(SEP)
        s1.extend(s2)
        print(s1)

        break # testing 1 sentence pair
    """

    dataset = Word2VecDataset(dev_data_path, VOCAB_SIZE, UNK, SEP, window_size=5)