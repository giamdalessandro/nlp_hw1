import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple, Any, Dict
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

from utils_classifier import BaseMLPClassifier, RecurrentLSTMClassifier, load_saved_model, \
                             train_evaluate, rnn_collate_fn
from utils_dataset import WiCDDataset, load_pretrained_embedding, indexify
from utils_aggregation import EmbAggregation

######################### Main test #######################
from torch import cuda
DEVICE = "cuda" if cuda.is_available() else "cpu"

DEV_PATH   = "data/dev.jsonl"
TRAIN_PATH = "data/train.jsonl"
SAVE_PATH  = "model/train/"

UNK = "UNK"
SEP = "SEP"
PAD = "PAD"

VOCAB_SIZE = 18000
NUM_EPOCHS = 70
BATCH_SIZE = 32

# APPROACH is set to 'wordEmb' to test the first hw approach, 'rnn' to test the second
APPROACH = "load"
PLOT = False   
print("\n################## my_stuff test code ################")

#torch.set_default_tensor_type('torch.cuda.FloatTensor')
print("Current cuda device ->", torch.cuda.get_device_name(torch.cuda.current_device()))

if APPROACH == "wordEmb":
    # create Dataset instance to handle training data
    train_dataset = WiCDDataset(TRAIN_PATH, UNK, SEP, vocab_size=VOCAB_SIZE, merge=True)

    #load pre-trained GloVe embeddings
    pretrained_emb, _ = load_pretrained_embedding(train_dataset.word_to_idx)
    emb_to_aggregation = EmbAggregation(pretrained_emb)

    train_dataset.preprocess_data(emb_to_aggregation, unk_token=UNK, sep_token=SEP)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    # create Dataset instance to handle dev data
    dev_dataset = WiCDDataset(DEV_PATH, UNK, SEP, merge=True, word_to_idx=train_dataset.word_to_idx, dev=True)
    dev_dataset.preprocess_data(emb_to_aggregation, unk_token=UNK, sep_token=SEP)
    dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)

    # instanciate NN classifier   
    sample_dim = train_dataset.get_sample_dim()[0]  # get actual sample dimension
    my_model = BaseMLPClassifier(input_features=sample_dim)
    optimizer = SGD(my_model.parameters(), lr=0.2, momentum=0.001)

    print("\n[INFO]: Beginning basic WordEmb training ...")
    print(f"[INFO]: {NUM_EPOCHS} epochs on device {DEVICE}.\n")
    history = train_evaluate( 
        model=my_model.cuda() if DEVICE == "cuda" else my_model,
        train_dataloader=train_dataloader,
        valid_dataloader=dev_dataloader,
        optimizer=optimizer,
        epochs=NUM_EPOCHS,
        device=DEVICE
    )

elif APPROACH == "rnn":
    # create Dataset instance to handle training data
    train_dataset = WiCDDataset(TRAIN_PATH, UNK, SEP, vocab_size=VOCAB_SIZE, merge=True)

    #load pre-trained GloVe embeddings
    pretrained_emb, _ = load_pretrained_embedding(train_dataset.word_to_idx)

    train_dataset.preprocess_data(unk_token=UNK, sep_token=SEP, rnn=True)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=rnn_collate_fn)

     # create Dataset instance to handle dev data
    dev_dataset = WiCDDataset(DEV_PATH, UNK, SEP, merge=True, word_to_idx=train_dataset.word_to_idx, dev=True)
    dev_dataset.preprocess_data(unk_token=UNK, sep_token=SEP, rnn=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, collate_fn=rnn_collate_fn)

    my_model = RecurrentLSTMClassifier(pretrained_emb=pretrained_emb)
    optimizer = SGD(my_model.parameters(), lr=0.2, momentum=0.001)

    print("\n[INFO]: Beginning RNN training ...")
    print(f"[INFO]: {NUM_EPOCHS} epochs on device {DEVICE}.\n")
    history = train_evaluate( 
        model=my_model,
        train_dataloader=train_dataloader,
        valid_dataloader=dev_dataloader,
        optimizer=optimizer,
        epochs=NUM_EPOCHS,
        early_stopping=True,
        rnn=True,
        device="cpu"
    )

else:

    spair = {"id": "dev.0", "lemma": "superior", "pos": "NOUN", "sentence1": "No clause in a contract shall be interpreted as evading the responsibility of superiors under international law.", "sentence2": "While fully aware that bishops and major superiors of religious institutes do not act as representatives or delegates of the Roman Pontiff, the Committee notes that subordinates in Catholic religious orders are bound by obedience to the Pope, in accordance with Canons 331 and 590 of the Code of canon Law.", "start1": "78", "end1": "87", "start2": "41", "end2": "50", "label": "False"}

    # create Dataset instance to handle training data
    train_dataset = WiCDDataset(TRAIN_PATH, UNK, SEP, vocab_size=VOCAB_SIZE, merge=True)

    #load pre-trained GloVe embeddings
    pretrained_emb, _ = load_pretrained_embedding(train_dataset.word_to_idx)

    my_model = load_saved_model(
        save_path="model/bests/rnn70_50d_nolemma_nostop_sub_1biRNN_local.pt",
        pretrained_emb=pretrained_emb
    )

    model_input = indexify(spair, word_to_idx=train_dataset.word_to_idx)

#print("\n[INFO]: model params ...")
#for k,v in my_model.state_dict().items():
#    print(f"\t {k}: {v.size()}")

# save trained model
torch.save(my_model.state_dict(), os.path.join(SAVE_PATH, f"rnn{NUM_EPOCHS}_glove50d_lemma_abssub_2lstm.pt"))

if PLOT:
    # plot loss and accuracy to inspect the training
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(np.arange(len(history["train_acc"])), history["train_acc"], label="train accuracy")
    axs[0].plot(np.arange(len(history["eval_acc"])), history["eval_acc"], label="val accuracy")
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("score")
    axs[0].set_xticks(np.arange(0,len(history["eval_acc"])+1,5))
    axs[0].grid()
    axs[0].legend()
    axs[0].set_title("Training history")

    axs[1].plot(np.arange(len(history["train_loss"])), history["train_loss"], color="green", label="train loss")
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("score")
    axs[1].set_xticks(np.arange(0,len(history["train_loss"])+1,5))
    axs[1].grid()
    axs[1].legend()

    plt.show()