import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple, Any, Dict
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

from utils_classifier import FooClassifier, load_saved_model, train_evaluate
from utils_dataset import WordEmbDataset, load_pretrained_embedding
from utils_aggregation import EmbAggregation

######################### Main test #######################
from torch import cuda
DEVICE = "cuda" if cuda.is_available() else "cpu"

DEV_PATH   = "data/dev.jsonl"
TRAIN_PATH = "data/train.jsonl"
SAVE_PATH  = "model/train/"

UNK = "UNK"
SEP = "SEP"
VOCAB_SIZE = 10000
NUM_EPOCHS = 70
BATCH_SIZE = 32


print("\n################## my_stuff test code ################")

#torch.set_default_tensor_type('torch.cuda.FloatTensor')
print("Current cuda device ->", torch.cuda.get_device_name(torch.cuda.current_device()))

# create Dataset instance to handle training data
train_dataset = WordEmbDataset(TRAIN_PATH, UNK, SEP, vocab_size=VOCAB_SIZE, merge=True)

#load pre-trained GloVe embeddings
pretrained_emb, _ = load_pretrained_embedding(train_dataset.word_to_idx)
emb_to_aggregation = EmbAggregation(pretrained_emb)

train_dataset.preprocess_data(emb_to_aggregation, unk_token=UNK, sep_token=SEP)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

# create Dataset instance to handle dev data
dev_dataset = WordEmbDataset(DEV_PATH, UNK, SEP, merge=True,
                                word_to_idx=train_dataset.word_to_idx, dev=True)
dev_dataset.preprocess_data(emb_to_aggregation, unk_token=UNK, sep_token=SEP)
dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)

# instanciate NN classifier   
sample_dim = train_dataset.get_sample_dim()[0]  # get actual sample dimension
my_model = FooClassifier(input_features=sample_dim, hidden_size=64, output_classes=1)
optimizer = SGD(my_model.parameters(), lr=0.3, momentum=0.001)

print("\n[INFO]: Beginning training ...\n")
history = train_evaluate( 
    model=my_model.cuda() if DEVICE == "cuda" else my_model,
    train_dataloader=train_dataloader,
    valid_dataloader=dev_dataloader,
    optimizer=optimizer,
    epochs=NUM_EPOCHS,
    device=DEVICE
)

# save trained model
torch.save(my_model.state_dict(), os.path.join(SAVE_PATH, f"train_eval{NUM_EPOCHS}_glove{sample_dim//2}d.pt"))

# plot loss and accuracy to inspect the training
fig = plt.figure()
plt.plot(np.arange(NUM_EPOCHS), history["train_acc"], label="train accuracy")
plt.plot(np.arange(NUM_EPOCHS), history["eval_acc"], label="val accuracy")
plt.xlabel("epoch")
plt.ylabel("score")
plt.xticks(np.arange(0,NUM_EPOCHS+1,10))

plt.title("Training history")
plt.grid()
plt.legend()
plt.show()