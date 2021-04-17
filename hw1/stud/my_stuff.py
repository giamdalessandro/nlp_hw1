import os
import re
import time
import jsonlines
import numpy as np
import collections

#from tqdm import tqdm
from typing import List, Tuple, Any, Dict
from torch import Tensor, FloatTensor, relu, softmax
from torch.nn import Embedding, Module, Linear, BCELoss
from torch.utils.data import Dataset


###########################################################
####           My classes & functions                  ####
###########################################################
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
        #self.window_size = window_size # [[w1,s1, w2,s1, ..., w|s1|,s1], ..., [w1,sn, ..., w|sn|,sn]] 
        self.unk_token   = unk_token
        self.sep_token   = sep_token
        self.data_json   = self.read_dataset(data_path)  # tuple(s_pairs,labels)
        self.build_vocabulary(vocab_size, unk_token, sep_token, merge=merge)

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

class FooClassifier(Module):
    """ TODO
    Classifier class.
    """
    def __init__(self, input_features: int, hidden_size: int, output_classes: int):
        super().__init__()
        self.hidden_layer = Linear(input_features, hidden_size)
        self.output_layer = Linear(hidden_size, output_classes)
        self.loss_fn = BCELoss()
        self.global_epoch = 0

    def forward(self, x: Tensor, y: Tensor) -> Dict[str, Tensor]:
        hidden_output = self.hidden_layer(x)
        hidden_output = relu(hidden_output)
        
        logits = self.output_layer(hidden_output).squeeze(1)
        probabilities = softmax(logits, dim=-1)
        result = {'logits': logits, 'probabilities': probabilities}

        # compute loss
        if y is not None:
            loss = self.loss(logits, y)
            result['loss'] = loss
        return result

    def loss(self, pred, y):
        return self.loss_fn(pred, y)

# TODO
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
    return embedding_mat, word_to_embedding

def embedding_lookUp(pretrained_emb: np.ndarray):
    num_embeddings = pretrained_emb.shape[0]
    embedding_dim  = pretrained_emb.shape[1]
    return Embedding(num_embeddings, embedding_dim).from_pretrained(FloatTensor(pretrained_emb))

def indexify():
    return NotImplementedError


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
    
    dev_data_path = DEV_PATH
    pretrained_path = os.path.join(PRETRAINED_DIR, "glove.6B", "glove.6B.50d.txt")

    dataset = WordEmbDataset(dev_data_path, VOCAB_SIZE, UNK, SEP, merge=False)
    pretrained_emb, _ = load_pretrained_embedding(pretrained_path, dataset.word_to_idx)

    # create pytorch embedding module
    num_embeddings = pretrained_emb.shape[0]
    embedding_dim  = pretrained_emb.shape[1]
    embeddings = Embedding(num_embeddings, embedding_dim).from_pretrained(FloatTensor(pretrained_emb))
    
    print(type(pretrained_emb))
    print(pretrained_emb.shape)
    #print("Embedding example:", vocab_emb["cat"])