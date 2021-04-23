import os
import matplotlib.pyplot as plt

from typing import List, Tuple, Any, Dict
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

from utils_classifier import FooClassifier, train_loop
from utils_dataset import WordEmbDataset, load_pretrained_embedding
from utils_aggregation import EmbAggregation

######################### Main test #######################
from torch import cuda
DEVICE = 'cuda' if cuda.is_available() else 'cpu'

DEV_PATH   = "data/dev.jsonl"
TRAIN_PATH = "data/train.jsonl"

UNK = "UNK"
SEP = "SEP"
VOCAB_SIZE = 10000
NUM_EPOCHS = 50
BATCH_SIZE = 32


if __name__ == '__main__':
    print("\n################## my_stuff test code ################")
    
    data_path = TRAIN_PATH

    dataset = WordEmbDataset(data_path, VOCAB_SIZE, UNK, SEP, merge=True)
    sample_dim = dataset.get_sample_dim()[0]
    print("Sample dim:", sample_dim)
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    
    my_model = FooClassifier(input_features=sample_dim, hidden_size=128, output_classes=1)
    optimizer = SGD(my_model.parameters(), lr=0.3, momentum=0.0)

    
    print("\n[INFO]: Beginning training ...\n")
    history = train_loop(
        model=my_model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        epochs=NUM_EPOCHS
    )

    