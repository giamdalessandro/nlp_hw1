import numpy as np
from typing import List, Tuple, Dict

from model import Model

def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
            
    # debug print
    print("[OK]: Module working.")

    return RandomBaseline() # StudentModel()


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

from utils_classifier import RecurrentLSTMClassifier, load_saved_model
from utils_dataset import WiCDDataset, load_pretrained_embedding

class StudentModel(Model):
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def predict(self, sentence_pairs: List[Dict]) -> List[str]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of sentences!

        # create WiCDDataset instance to setup the vocabulary
        train_dataset = WiCDDataset(TRAIN_PATH, UNK, SEP, vocab_size=VOCAB_SIZE, merge=True)

        #load pre-trained GloVe.6B embeddings
        pretrained_emb, _ = load_pretrained_embedding(train_dataset.word_to_idx)
        
        my_model = load_saved_model(
            save_path="model/bests/rnn70_50d_nolemma_nostop_sub_1biRNN_local.pt",
            pretrained_emb=pretrained_emb
        )
        pass
