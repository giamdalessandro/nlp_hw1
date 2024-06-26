import os
import re
import time
import collections
import jsonlines
import numpy as np

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import torch
from torch import Tensor, LongTensor
from torch.nn import Embedding, Module
from torch.utils.data import Dataset, DataLoader

try:
    from utils_aggregation import EmbAggregation, embedding_lookUp
except:
    from stud.utils_aggregation import EmbAggregation, embedding_lookUp

PRETRAINED_DIR  = "./model/pretrained_emb/"
PRETRAINED_EMB  = "glove.6B"
PRETRAINED_FILE = os.path.join(PRETRAINED_DIR, PRETRAINED_EMB, "glove.6B.50d.txt")


def merge_pair(spair: dict, sep_token: str, separate: bool=False):
    """
    Merge the sentences of a pair into one context, where the two are separated 
    by the sep_token.
    """
    s1 = spair["sentence1"].strip().split(" ")
    s2 = spair["sentence2"].strip().split(" ")
    if separate:
        s1.append(sep_token)
    s1.extend(s2)
    s1.append(spair["lemma"])

    return s1

def load_pretrained_embedding(word_to_idx: dict, path: str=PRETRAINED_FILE, pad_token: str="<PAD>"):
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
            if word == pad_token:
                # Add a zeros vector as the pad token embedding
                token_emb = np.zeros(shape=(emb_dim, )) 
                word_to_embedding[word] = token_emb
                embedding_list.append(token_emb)
            else:    
                # if word is not in GloVe, adds a random tensor as its embedding;
                # it does so also for both unknown and separator tokens
                token_emb = np.random.normal(scale=0.6, size=(emb_dim, ))  #torch.rand((emb_dim, )) 
                word_to_embedding[word] = token_emb
                embedding_list.append(token_emb)
                #print("missing word:", word)
            missing += 1

    distinct_words = len(word_to_idx)
    print(f"Total of missing embeddings: {missing} ({round(missing/distinct_words,6)*100}% of vocabulary)" )

    embedding_mat = np.array(embedding_list, dtype=np.float64)
    print(f"Total embeddings: ({len(embedding_list)},{emb_dim})")
    return embedding_mat #, emb_dim

def indexify(
        spair: dict, 
        word_to_idx: dict, 
        unk_token: str, 
        sep_token: str, 
        stopwords,
        stop: bool=True,
        lemma_first: bool=False,
        lemma_last: bool=False, 
        rnn: bool=False
    ):
    """
    Maps the words of the input sentences pair to the matching vocabulary indexes. 
        - lemma_first: whether to push the lemma at the beginning of the sentence;
        - stop: whether to remove the stopwords in 'stopWords' from the sentence;
    """
    # get lemma vocabulary index
    try:
        lemma_idx = word_to_idx[spair["lemma"]]
    except KeyError as e:    
        lemma_idx = word_to_idx[unk_token]

    # get target words in each sentence of the pair
    target_s1 = spair["sentence1"][int(spair["start1"]):int(spair["end1"])]
    target_s2 = spair["sentence2"][int(spair["start2"]):int(spair["end2"])]
    
    s1_indexes = []
    for word in spair["sentence1"].strip().split():
        if word == target_s1:
            # lemmatization of target word
            word = spair["lemma"]  
            if rnn:
                continue
        elif word in stopwords and stop:
            # skip word if it's a stopword  
            continue
        try:
            s1_indexes.append(word_to_idx[word])
        except KeyError as e:   
            s1_indexes.append(word_to_idx[unk_token])
    
    s2_indexes = []
    for word in spair["sentence2"].strip().split():
        if word == target_s2:
            # lemmatization of target word
            word = spair["lemma"]
            if rnn:
                continue
        elif word in stopwords and stop:
            # skip word if it's a stopwords 
            continue
        try:
            s2_indexes.append(word_to_idx[word])
        except KeyError as e:    
            s2_indexes.append(word_to_idx[unk_token])
    
    if lemma_first:
        s1_idx = [lemma_idx]
        s1_indexes.extend(s1_idx)
        s2_idx = [lemma_idx]
        s2_indexes.extend(s2_idx)
    elif lemma_last:
        s1_indexes.append(lemma_idx)
        s2_indexes.append(lemma_idx)

    if not rnn:
        return s1_indexes, s2_indexes, lemma_idx
    #elif merge:
    #    s1_indexes.extend(s2_indexes)
    #    return s1_indexes
    else:
        return Tensor(s1_indexes), Tensor(s2_indexes)
    

class WiCDDataset(Dataset):
    """ TODO override __getItem__()
    Class to manage the dataset and to properly load pretrained embeddings 
    (subclass of torch.util.data.Dataset).
    """
    def __init__(self, 
        data_path: str, 
        unk_token: str, 
        sep_token: str, 
        pad_token: str,
        merge: bool,
        vocab_size: int=10000, 
        word_to_idx: dict=None, 
        dev: bool=False
        ):
        """
        Args:
            - data_path   : Path to the dataset file;
            - vocab_size  : Maximum amount of words that we want to embed;
            - unk_token   : token to represent unknown words;
            - window_size : Number of words to consider as context.
        """
        self.unk_token   = unk_token
        self.sep_token   = sep_token
        self.pad_token   = pad_token
        self.word_to_idx = word_to_idx  # passed when creating dev data module
        self.data_json = self.__read_dataset(data_path)  
        if not dev:
            # Build the vocabolary that will be used for training, and initialize some useful data structures
            self.__build_vocabulary(vocab_size, unk_token, sep_token, pad_token, merge=merge)

    def __tokenize_line(self, line: str, pattern='\W'):
        """
        Tokenizes a single line (e.g. "The pen is on the table" -> 
        ["the, "pen", "is", "on", "the", "table"]).
        """
        #tokens = nltk.word_tokenize(line)
        res = ""
        for word in re.split(pattern, line.lower()):
            if word:
                res += " " + word.lower()
        return res

    def __read_dataset(self, data_path: str):
        """
        Reads the dataset and converts each sentence in the input file into a list of tokenized words.
        """       
        print(f"\n[INFO]: Loading data from '{data_path}'...")
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

    def __build_vocabulary(self, vocab_size: int, unk_token: str, sep_token: str, pad_token: str, merge: bool):
        """ TODO
        Defines the vocabulary to be used. Builds a mapping (word, index) for
        each word in the vocabulary. It adds the following attributes to the class: 
            self.distinct_words   # number of distinct words
            self.word_to_idx      # (word, idx) mapping
            self.frequency        # (word, freq) mapping
            self.tot_occurrences  # number of total words
            self.id2word          # (idx, word) mapping
            self.data_samples     # (spair, label) mapping

        Args:
            - vocab_size: size of the vocabolary;
            - unk_token : token to associate with unknown words;
            - merge     : Wheter to merge the sentence pairs into one;
            - sep_token : token to separate sentence pairs (only needed when 
                'merge' is set True).
        """
        print("\n[INFO]: building vocabulary ...")
        counter_list = []
        lemma_list = []
        for spair in self.data_json[0]:
            # context is a list of tokens within a single sentence
            context = merge_pair(spair=spair, sep_token=sep_token)
            counter_list.extend(context)
            lemma_list.append(context[-1])
            
        counter = collections.Counter(counter_list)
        self.distinct_words = len(counter)
        print(f"Number of distinct words: {len(counter)}")

        # consider only the (vocab size - 1) most common words to build the vocab
        size_considered = (vocab_size - 1) if not merge else (vocab_size - 2)
        most_common = enumerate(counter.most_common(size_considered))
        dictionary = {key: index for index, (key, _) in most_common}

        assert unk_token not in dictionary
        dictionary[unk_token] = size_considered
        if merge:
            dictionary[sep_token] = vocab_size - 1
            
        dictionary[pad_token] = vocab_size
        self.word_to_idx = dictionary

        # dictionary with (word, frequency) pairs -- including only words that are in the vocabulary
        dict_counts = {x: counter[x] for x in dictionary if (x != unk_token and x != sep_token and x != pad_token)}
        self.frequency = dict_counts
        self.tot_occurrences = sum(dict_counts[x] for x in dict_counts)
        print(f"Total occurrences of words in dictionary: {self.tot_occurrences}")

        less_freq_word = min(dict_counts, key=counter.get)
        print(f"Less frequent word in dictionary appears {dict_counts[less_freq_word]} times ({less_freq_word})")

        # index to word dictonary
        self.id2word = {value: key for key, value in dictionary.items()}
        return
        
    def preprocess_data(self, emb_to_aggregation: Module=None, unk_token: str="<UNK>", 
                        sep_token: str="<SEP>", rnn=False):
        """
        Preprocess the data to create data samples suitable for the classifier. 
        The samples are couples having a sentences pair associated with its 
        groundtruth label (e.g. <s_pair,label>).
   
            - unk_token : token to associate with unknown words;
            - sep_token : token to separate sentence pairs.
        """
        assert self.word_to_idx is not None
        stopWords = set(stopwords.words('english'))
        
        count = 0
        samples = []
        for spair, label in zip(self.data_json[0],self.data_json[1]):
            paragraph = indexify(spair, self.word_to_idx, unk_token, sep_token, stopWords, rnn=rnn)  
            
            # apply aggregation function if working only with embeddings,
            # otherwise just use the indexified sentences 
            aux_emb = emb_to_aggregation(paragraph) if not rnn else paragraph
            aux_label = 1 if label == "True" else 0
            sample = (aux_emb, aux_label)
            samples.append(sample)
            count += 1

        print(f"[INFO]: Loaded {count} samples.")
        self.data_samples = samples
        return

    def get_sample_dim(self, idx: int=0):
        """ Returns the idx-th sample pair dimensions, where a sample is a tuple (pair, label)"""
        return self.__getitem__(idx)[0].size()

    # overrided method
    def __len__(self):
        """ Returns the number of samples in our dataset """
        return len(self.data_samples)

    def __getitem__(self, idx):
        """ Returns the idx-th sample """
        return self.data_samples[idx]
