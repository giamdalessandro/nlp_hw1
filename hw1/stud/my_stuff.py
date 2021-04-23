import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple, Any, Dict
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

from utils_classifier import FooClassifier, train_loop, load_saved_model, evaluate_model
from utils_dataset import WordEmbDataset, load_pretrained_embedding
from utils_aggregation import EmbAggregation

######################### Main test #######################
from torch import cuda
DEVICE = 'cuda' if cuda.is_available() else 'cpu'

DEV_PATH   = "data/dev.jsonl"
TRAIN_PATH = "data/train.jsonl"
SAVE_PATH  = "model/train/"

UNK = "UNK"
SEP = "SEP"
VOCAB_SIZE = 10000
NUM_EPOCHS = 130
BATCH_SIZE = 32

DEV_VOCAB_SIZE = 5000


TRAIN = False
print("\n################## my_stuff test code ################")
if TRAIN:    
    train_dataset = WordEmbDataset(TRAIN_PATH, VOCAB_SIZE, UNK, SEP, merge=True)
    sample_dim = train_dataset.get_sample_dim()[0]
    print("Train sample dim:", sample_dim)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    
    my_model = FooClassifier(input_features=sample_dim, hidden_size=128, output_classes=1)
    optimizer = SGD(my_model.parameters(), lr=0.3, momentum=0.0)

    
    print("\n[INFO]: Beginning training ...\n")
    history = train_loop(
        model=my_model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        epochs=NUM_EPOCHS
    )

    # save trained model
    torch.save(my_model.state_dict(), os.path.join(SAVE_PATH, f"train{NUM_EPOCHS}_glove{sample_dim//2}d.pt"))

    # plot loss and accuracy to inspect the training
    fig = plt.figure()
    plt.plot(np.arange(NUM_EPOCHS), history["accuracy"], label="accuracy")
    plt.plot(np.arange(NUM_EPOCHS), history["loss"], label="loss")
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.xticks(np.arange(0,NUM_EPOCHS+1,10))
    
    plt.title("Training history")
    plt.grid()
    plt.legend()
    plt.show()

################### test saved model with dev data
# create test Dataset instance
test_dataset = WordEmbDataset(DEV_PATH, VOCAB_SIZE, UNK, SEP, merge=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

test_model = load_saved_model(os.path.join(SAVE_PATH, "train130_glove100d.pt"))
print("\n[INFO]: Beginning testing ...\n")
