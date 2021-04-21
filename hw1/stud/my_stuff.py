import os

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
    
    data_path = DEV_PATH

    dataset = WordEmbDataset(data_path, VOCAB_SIZE, UNK, SEP, merge=False)

    pretrained, _ = load_pretrained_embedding(dataset.word_to_idx)
    e = EmbAggregation(pretrained)

    print(e(dataset.data_samples[0][0]))
    """
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    my_model = FooClassifier(input_features=50, hidden_size=128, output_classes=1)
    optimizer = SGD(my_model.parameters(), lr=0.2, momentum=0.0)

    print("\n[INFO]: Beginning training ...\n")
    history = train_loop(
        model=my_model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        epochs=NUM_EPOCHS
    )"""