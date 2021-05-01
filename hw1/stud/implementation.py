import numpy as np
from typing import List, Tuple, Dict

from model import Model

def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
            
    # debug print
    print("[OK]: Module working.")

    return StudentModel() # RandomBaseline() 


class RandomBaseline(Model):

    options = [
        ('True', 40000),
        ('False', 40000),
    ]

    def __init__(self):

        self._options = [option[0] for option in self.options]
        self._weights = np.array([option[1] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, sentence_pairs: List[Dict]) -> List[Dict]:
        return [str(np.random.choice(self._options, 1, p=self._weights)[0]) for x in sentence_pairs]


######################################
# Function and classes needed to load the pre-trained model and build
# the vocabulary 
import torch
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from stud.utils_classifier import RecurrentLSTMClassifier, load_saved_model, test_collate_fn
from stud.utils_dataset import WiCDDataset, load_pretrained_embedding, indexify

class StudentModel(Model):
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    def __init__(self):
        self.train_path = "model/train/train.jsonl"
        self.unk_token  = "UNK"
        self.sep_token  = "SEP"
        self.pad_token  = "PAD"
        self.vocab_size = 18000
        self._step_size = 32

    def predict(self, sentence_pairs: List[Dict]) -> List[str]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of sentences!

        # create Dataset instance to handle training data
        train_dataset = WiCDDataset(
            data_path=self.train_path, 
            unk_token=self.unk_token, 
            sep_token=self.sep_token, 
            vocab_size=self.vocab_size, 
            merge=True
        )
        # load pre-trained GloVe embeddings
        pretrained_emb, _ = load_pretrained_embedding(train_dataset.word_to_idx)
        # load stopwords from nltk
        stopWords = set(stopwords.words('english'))

        res = [] # list that will contain the classification results (e.g. ['True','False', ..., 'False'])
        for step in range(0,len(sentence_pairs), self._step_size):
            if step+32 <= len(sentence_pairs):
                step_pairs = sentence_pairs[step:step+self._step_size]
            else:
                step_pairs = sentence_pairs[step:]

            data_elements = []
            for pair in step_pairs:
                elem = indexify(
                    spair=pair, 
                    word_to_idx=train_dataset.word_to_idx,
                    unk_token=self.unk_token,
                    sep_token=self.sep_token,
                    stopwords=stopWords,
                    rnn=True      
                )
                data_elements.append(elem)
                

            my_model = load_saved_model(
                save_path="model/bests/rnn70_glove50d_lemmalast_abssub_1lstm.pt",
                pretrained_emb=pretrained_emb
            )

            x, xlens = test_collate_fn(data_elements)
            with torch.no_grad():
                out = my_model(x, xlens)
                step_res = [round(i) for i in out["probabilities"].detach().numpy()]
                step_res = ["True" if i == 1 else "False" for i in step_res]
                res.extend(step_res)

        return res
