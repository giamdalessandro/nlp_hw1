import os
import re
import time
import jsonlines
import numpy as np
import collections

#from tqdm import tqdm
from torch.utils.data import IterableDataset
from typing import List, Tuple, Any, Dict

"""
My classes & functions
"""
class Word2VecDataset(IterableDataset):
    # TODO: change class name
    def __init__(self, data_path: str, vocab_size: int, unk_token: str, sep_token: str, 
                       window_size: int, merge: bool):
        """
        Args:
            - data_path   : Path to the dataset file;
            - vocab_size  : Maximum amount of words that we want to embed;
            - unk_token   : token to represent unknown words;
            - window_size : Number of words to consider as context.
        """
        self.unk_token   = unk_token
        self.sep_token   = sep_token
        self.window_size = window_size # [[w1,s1, w2,s1, ..., w|s1|,s1], ..., [w1,sn, ..., w|sn|,sn]] 
        self.data_json   = self.read_dataset(data_path)  # tuple(s_pairs,labels)
        self.build_vocabulary(vocab_size, unk_token, sep_token, merge=merge)

    # TODO
    def __iter__(self):
        """
        Overwrites the __iter__() method of the superclass (torch.utils.data.IterableDataset).
        """
        sentence_pairs = self.data_json[0]
        for spair in sentence_pairs:
            if merge:  #RABARBARO
                sentence = self.merge_pair(spair=spair, sep_token=sep_token)
            #else:
            #    sentence  = spair["sentence1"]
            #    sentence2 = spair["sentence2"]
            #    sentence.extend(sentence2)
                
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

    def keep_word(self,  word: str):
        '''
        Implements negative sampling and returns true if we can keep the occurrence as training instance.
        '''
        z = self.frequency[word] / self.tot_occurrences
        p_keep = np.sqrt(z / 10e-3) + 1
        p_keep *= 10e-3 / z # higher for less frequent instances
        return np.random.rand() < p_keep # toss a coin and compare it to p_keep to keep the word

    def merge_pair(self, spair: dict, sep_token: str):
        """
        Merge the sentences of a pair into one context, where the two are separated 
        by the sep_token. # may include this in tokenize_line #RABARBARO
        """
        s1 = spair["sentence1"]
        s2 = spair["sentence2"]
        
        s1.append(sep_token)
        s1.extend(s2)
        return s1

    def tokenize_line(self, line: str, pattern='\W'):
        """
        Tokenizes a single line (e.g. "The pen is on the table" -> 
        ["the, "pen", "is", "on", "the", "table"]).
        """
        return [word.lower() for word in re.split(pattern, line.lower()) if word]

    def read_dataset(self, data_path: str):
        """
        Reads the dataset and converts each sentence in the input file into a list of tokenized words.
        """       
        print(f"[INFO]: Loading data from '{data_path}'...")
        sentence_pairs = []
        labels = []

        with jsonlines.open(data_path) as f:
            for obj in f:
                labels.append(obj.pop('label'))

                obj["sentence1"] = self.tokenize_line(obj["sentence1"])
                obj["sentence2"] = self.tokenize_line(obj["sentence2"])
                sentence_pairs.append(obj)

        assert len(sentence_pairs) == len(labels)

        print("labels:        ",len(labels))
        print("sentence pairs:",len(sentence_pairs))
        print("[INFO]: data loaded successfully.")
        return sentence_pairs, labels

    # TODO
    def build_vocabulary(self, vocab_size: int, unk_token: str, sep_token: str, merge: bool):
        """ TODO
        Defines the vocabulary to be used. Builds a mapping (word, index) for
        each word in the vocabulary.

        Args:
            - vocab_size: size of the vocabolary;
            - unk_token : token to associate with unknown words;
            - merge     : Wheter to merge the sentence pairs into one;
            - sep_token : token to separate sentence pairs (only needed when 
                'merge' is set True).
        """
        print("\n[INFO]: building vocabulary ...")
        counter_list = []
        # context is a list of tokens within a single sentence
        for spair in self.data_json[0]:
            if merge:
                context = self.merge_pair(spair=spair, sep_token=sep_token)
                counter_list.extend(context)

            else:
                context1 = spair["sentence1"]
                context2 = spair["sentence2"]
                counter_list.extend(context1)
                counter_list.extend(context2)
            
        counter = collections.Counter(counter_list)
        self.distinct_words = len(counter)
        print(f"Number of distinct words: {len(counter)}")

        # consider only the (vocab size -1) most common words to build the vocab
        most_common = enumerate(counter.most_common(vocab_size - 1))
        dictionary = {key: index for index, (key, _) in most_common}
        # all the other words are mapped to UNK
        assert unk_token not in dictionary
        dictionary[unk_token] = vocab_size - 1
        if merge:
            dictionary[unk_token] = vocab_size - 1
            dictionary[sep_token] = vocab_size

        self.word2id = dictionary

        # dictionary with (word, frequency) pairs -- including only words that are in the vocabulary
        dict_counts = {x: counter[x] for x in dictionary if x is not unk_token}
        self.frequency = dict_counts
        self.tot_occurrences = sum(dict_counts[x] for x in dict_counts)

        print(f"Total occurrences of words in dictionary: {self.tot_occurrences}")

        less_freq_word = min(dict_counts, key=counter.get)
        print(f"Less frequent word in dictionary appears {dict_counts[less_freq_word]} times ({less_freq_word})")

        # index to word dictonary
        self.id2word = {value: key for key, value in dictionary.items()}

        # data is the text converted to indexes, as list of lists
        data = []
        # for each sentence
        for spair in self.data_json[0]:
            #sentence = self.merge_pair(spair=spair, sep_token=sep_token)
            if merge:  #RABARBARO
                sentence = self.merge_pair(spair=spair, sep_token=sep_token)
            else:
                sentence  = spair["sentence1"]
                sentence2 = spair["sentence2"]
                sentence.extend(sentence2)

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
    def load_pretrained_embedding(self, path: str):
        """
        Loads pre-trained word embeddings from 'path' file, and retrieves an 
        embeddings matrix associating each vocabulary word to its corresponding
        pre-trained embedding, if any. 

            - path    : file path were embeddings are stored.
        """
        if not os.path.isfile(path):
            print(f"[INFO]: embedding file not found in: {path} ...")
            exit(1)

        # load the pre-trained embeddings from file.
        print(f"\n[INFO]: loading embedding from '{path}' ...")
        pretrained_embs = {}
        tot_words = 0
        with open(path, "r") as f:
            for row in f.readlines():
                row_list = row.strip().split(" ")
                if tot_words == 0:
                    emb_dim = len(row_list) - 1
                    #assert len(row_list) - 1 == emb_dim

                pretrained_embs[row_list[0]] = row_list[1:] 
                tot_words += 1

        print(f"loaded {len(pretrained_embs)} pre-trained embeddings of dim {emb_dim} ...")

        # Build a dictionary mapping vocabulary words to the relative pre-trained embeddings.
        # Map the word indexes to the corresponding embedding, to create
        # the actual embedding matrix.
        vocab_embeddings = {}
        embedding_list = []
        missing = 0

        sorted_w2id = dict(sorted(self.word2id.items(), key=lambda item: item[1]))
        for word, idx in sorted_w2id.items():
            try: 
                vocab_embeddings[word] = pretrained_embs[word]
                embedding_list.append(pretrained_embs[word])

            except KeyError as e:
                if word == self.unk_token or word == self.sep_token:
                    token_emb = np.random.rand(emb_dim)
                    vocab_embeddings[word] = token_emb
                    embedding_list.append(token_emb)
                else:
                    #print("missing word:", word)
                    missing += 1

        embedding_mat = np.array(embedding_list, dtype=np.float64)
        print(f"Total of missing embeddings: {missing} ({round(missing/self.distinct_words,6)*100}% of vocabulary)" )

        return embedding_mat, vocab_embeddings


######################### Main ########################################
PRETRAINED_DIR = "./model/pretrained_emb/"
DEV_PATH   = "data/dev.jsonl"
TRAIN_PATH = "data/train.jsonl"

UNK = "UNK"
SEP = "SEP"
VOCAB_SIZE = 30000

if __name__ == '__main__':
    print("\n################## my_stuff test code ################")
    
    dev_data_path = DEV_PATH
    pretrained_path = os.path.join(PRETRAINED_DIR, "glove.6B", "glove.6B.100d.txt")

    dataset = Word2VecDataset(dev_data_path, VOCAB_SIZE, UNK, SEP, window_size=5, merge=False)
    pretrained_emb, vocab_emb = dataset.load_pretrained_embedding(pretrained_path)

    print(pretrained_emb.shape)
    print("Embedding example:", vocab_emb["cat"])