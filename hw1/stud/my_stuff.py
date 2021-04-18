import os
import re
import time
import jsonlines
import numpy as np
import collections

#from tqdm import tqdm
from typing import List, Tuple, Any, Dict

import torch
from torch import Tensor, LongTensor
from torch.nn import Embedding, Module
from torch.utils.data import Dataset, DataLoader

from panichetto import FooClassifier, train_loop

###########################################################
####           My classes & functions                  ####
###########################################################
def merge_pair(spair: dict, sep_token: str, separate: bool):
    """
    Merge the sentences of a pair into one context, where the two are separated 
    by the sep_token.
    """
    s1, s2 = spair["sentence1"], spair["sentence2"]
    if separate:
        s1.append(sep_token)
    s1.extend(s2)
    #print(s1)
    return s1

def load_pretrained_embedding(path: str, word_to_idx: dict):
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
    word_to_pretrained = {}
    tot_words = 0
    with open(path, "r") as f:
        for row in f.readlines():
            row_list = row.strip().split(" ")
            if tot_words == 0:
                emb_dim = len(row_list) - 1
                #assert len(row_list) - 1 == emb_dim

            word_to_pretrained[row_list[0]] = row_list[1:] 
            tot_words += 1

    print(f"loaded {len(word_to_pretrained)} pre-trained embeddings of dim {emb_dim} ...")

    # Build a dictionary mapping vocabulary words to the relative pre-trained embeddings.
    # Map the word indexes to the corresponding embedding, to create
    # the actual embedding matrix.
    word_to_embedding = {}
    embedding_list = []
    missing = 0

    sorted_w2id = dict(sorted(word_to_idx.items(), key=lambda item: item[1]))
    for word, idx in sorted_w2id.items():
        try: 
            word_to_embedding[word] = word_to_pretrained[word]
            embedding_list.append(word_to_pretrained[word])

        except KeyError as e:
            #if word is not in GloVe, adds a random tensor as its embedding;
            # it does so also for both unknown and separator tokens
            token_emb = np.random.normal(scale=0.6, size=(emb_dim, ))
            word_to_embedding[word] = token_emb
            embedding_list.append(token_emb)
            #print("missing word:", word)
            missing += 1

    distinct_words = len(word_to_idx)
    print(f"Total of missing embeddings: {missing} ({round(missing/distinct_words,6)*100}% of vocabulary)" )

    embedding_mat = np.array(embedding_list, dtype=np.float64)
    print("Total embeddings:",embedding_mat.shape)
    return embedding_mat, word_to_embedding

def indexify(spair: dict, word_to_idx: dict, unk_token: str, sep_token: str):
    """
    Maps the words of the input sentences pair to the matching vocabulary indexes. 
    """
    indexes = []
    merged = merge_pair(spair, sep_token, False)
    for word in merged:
        try:
            indexes.append(word_to_idx[word])
        except KeyError as e:
            indexes.append(word_to_idx[unk_token])
    
    return indexes

def embedding_lookUp(pretrained_emb: np.ndarray):
    """
    Lookup table that matches a list of word indexes to their respective embedding tensors,
    creating a pytorch embedding module.
    """
    num_embeddings = pretrained_emb.shape[0]
    embedding_dim  = pretrained_emb.shape[1]
    return Embedding(num_embeddings, embedding_dim).from_pretrained(Tensor(pretrained_emb))

def dummy_aggreggation(sentence_embeddings):
    # compute the mean of the e,beddings of a sentence
    return torch.mean(sentence_embeddings, dim=0).float()

class WordEmbDataset(Dataset):
    """ TODO override __getItem__()
    Class to manage the dataset and to properly load pretrained embeddings 
    (subclass of torch.util.data.Dataset).
    """
    def __init__(self, data_path: str, vocab_size: int, unk_token: str, sep_token: str, merge: bool):
        """
        Args:
            - data_path   : Path to the dataset file;
            - vocab_size  : Maximum amount of words that we want to embed;
            - unk_token   : token to represent unknown words;
            - window_size : Number of words to consider as context.
        """
        self.unk_token    = unk_token
        self.sep_token    = sep_token
        self.data_json    = self.__read_dataset(data_path)  
        # Build the vocabolary that will be used fro training and other useful data structures
        self.__build_vocabulary(vocab_size, unk_token, sep_token, merge=merge)
        # Preprocess the dateset and provide the aggregated samples
        self.data_samples = self._preprocess_samples(unk_token, sep_token)

    def __tokenize_line(self, line: str, pattern='\W'):
        """
        Tokenizes a single line (e.g. "The pen is on the table" -> 
        ["the, "pen", "is", "on", "the", "table"]).
        """
        return [word.lower() for word in re.split(pattern, line.lower()) if word]

    def __read_dataset(self, data_path: str):
        """
        Reads the dataset and converts each sentence in the input file into a list of tokenized words.
        """       
        print(f"[INFO]: Loading data from '{data_path}'...")
        sentence_pairs = []
        labels = []

        with jsonlines.open(data_path) as f:
            for obj in f:
                labels.append(obj.pop('label'))

                obj["sentence1"] = self.__tokenize_line(obj["sentence1"])
                obj["sentence2"] = self.__tokenize_line(obj["sentence2"])
                sentence_pairs.append(obj)

        assert len(sentence_pairs) == len(labels)

        print("labels:        ",len(labels))
        print("sentence pairs:",len(sentence_pairs))
        print("[INFO]: data loaded successfully.")
        return sentence_pairs, labels

    def __build_vocabulary(self, vocab_size: int, unk_token: str, sep_token: str, merge: bool):
        """ TODO
        Defines the vocabulary to be used. Builds a mapping (word, index) for
        each word in the vocabulary. It adds the following attributes to the class: 
            self.distinct_words
            self.word_to_idx
            self.frequency
            self.tot_occurrences 
            self.id2word
            self.data_samples

        Args:
            - vocab_size: size of the vocabolary;
            - unk_token : token to associate with unknown words;
            - merge     : Wheter to merge the sentence pairs into one;
            - sep_token : token to separate sentence pairs (only needed when 
                'merge' is set True).
        """
        print("\n[INFO]: building vocabulary ...")
        counter_list = []
        for spair in self.data_json[0]:
            # context is a list of tokens within a single sentence
            context = merge_pair(spair=spair, sep_token=sep_token, separate=merge)
            counter_list.extend(context)
            
        counter = collections.Counter(counter_list)
        self.distinct_words = len(counter)
        print(f"Number of distinct words: {len(counter)}")

        # consider only the (vocab size -1) most common words to build the vocab
        most_common = enumerate(counter.most_common(vocab_size - 1))
        dictionary = {key: index for index, (key, _) in most_common}

        assert unk_token not in dictionary
        dictionary[unk_token] = vocab_size - 1
        if merge:
            dictionary[sep_token] = vocab_size
        self.word_to_idx = dictionary

        # dictionary with (word, frequency) pairs -- including only words that are in the vocabulary
        dict_counts = {x: counter[x] for x in dictionary if x is not unk_token}
        self.frequency = dict_counts
        self.tot_occurrences = sum(dict_counts[x] for x in dict_counts)
        print(f"Total occurrences of words in dictionary: {self.tot_occurrences}")

        less_freq_word = min(dict_counts, key=counter.get)
        print(f"Less frequent word in dictionary appears {dict_counts[less_freq_word]} times ({less_freq_word})")

        # index to word dictonary
        self.id2word = {value: key for key, value in dictionary.items()}
        return
        
    def _preprocess_samples(self, unk_token: str, sep_token: str):
        """
        Preprocess the data to create data samples. The samples are couples having 
        a sentence pair associated with its label (e.g. <s_pair,label>).
   
            - unk_token : token to associate with unknown words;
            - sep_token : token to separate sentence pairs.
        """
        # load pre-trained embeddings and create embedding module
        pretrained_emb, _ = load_pretrained_embedding(pretrained_path, self.word_to_idx)
        emb_lookup = embedding_lookUp(pretrained_emb)
        
        count = 0
        samples = []
        for spair, label in zip(self.data_json[0],self.data_json[1]):
            paragraph = indexify(spair, self.word_to_idx, unk_token, sep_token)  
            embs = emb_lookup(LongTensor(paragraph))
            # apply aggregation function
            aux_emb = dummy_aggreggation(embs)
            aux_label = 1 if label == "True" else 0
            sample = (aux_emb, aux_label)
            #print(sample)
            samples.append(sample)
            count += 1

        print(f"Loaded {count} samples.")
        return samples

    # overrided method
    def __len__(self):
        """ Returns the number of samples in our dataset """
        return len(self.data_samples)

    def __getitem__(self, idx):
        """ Returns the idx-th sample """
        return self.data_samples[idx]


######################### Main test #######################
#from torch import cuda
#DEVICE = 'cuda' if cuda.is_available() else 'cpu'

PRETRAINED_DIR = "./model/pretrained_emb/"
DEV_PATH   = "data/dev.jsonl"
TRAIN_PATH = "data/train.jsonl"

UNK = "UNK"
SEP = "SEP"
VOCAB_SIZE = 10000

if __name__ == '__main__':
    print("\n################## my_stuff test code ################")
    
    data_path = DEV_PATH
    pretrained_path = os.path.join(PRETRAINED_DIR, "glove.6B", "glove.6B.50d.txt")

    dataset = WordEmbDataset(data_path, VOCAB_SIZE, UNK, SEP, merge=False)
    train_dataloader = DataLoader(dataset, batch_size=32)

    my_model = FooClassifier(input_features=50, hidden_size=128, output_classes=1)
    optimizer = torch.optim.SGD(
        my_model.parameters(),
        lr=0.2,
        momentum=0.0
    )

    print("\n[INFO]: starting training ...")
    history = train_loop(
        model=my_model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        epochs=5
    )