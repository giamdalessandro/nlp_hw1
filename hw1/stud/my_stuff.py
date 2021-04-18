import os

from typing import List, Tuple, Any, Dict
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

from panichetto import FooClassifier, train_loop
from accaso import WordEmbDataset, load_pretrained_embedding


######################### Main test #######################
#from torch import cuda
#DEVICE = 'cuda' if cuda.is_available() else 'cpu'

DEV_PATH   = "data/dev.jsonl"
TRAIN_PATH = "data/train.jsonl"

UNK = "UNK"
SEP = "SEP"
VOCAB_SIZE = 10000

if __name__ == '__main__':
    print("\n################## my_stuff test code ################")
    
    data_path = DEV_PATH

    dataset = WordEmbDataset(data_path, VOCAB_SIZE, UNK, SEP, merge=False)
    train_dataloader = DataLoader(dataset, batch_size=32)

    my_model = FooClassifier(input_features=50, hidden_size=128, output_classes=1)
    optimizer = SGD(
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