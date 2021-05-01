import torch
import numpy as np

from torch import Tensor, LongTensor, load
from torch import relu, sigmoid
from torch.nn import Module, Linear, BCELoss, LSTM, Embedding
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score
from utils_aggregation import EmbAggregation, embedding_lookUp

VERBOSE = False

def load_saved_model(save_path: str, mode="rnn", pretrained_emb=None):
    """
    Loads a saved and pre-trained model, generated with the BaseMLPClassifier 
    or the RecurrentLSTMClassifier Module.
    """
    print(f"\n[INFO]: loading pre-trained model from '{save_path}'...")
    if mode == "rnn": 
        model = RecurrentLSTMClassifier(pretrained_emb=pretrained_emb)
    elif mode == "base":
        model = BaseMLPClassifier()

    # load model weights
    model.load_state_dict(load(save_path))
    # set model in evaluation mode
    model.eval()
    print(model.state_dict().keys())
    return model

@torch.no_grad()
def evaluate_accuracy(model: Module, dataloader: DataLoader):#
    y_true = []
    y_pred = []
    for x, y in dataloader:
        output = model(x.cuda(), y.cuda())
        predictions = output['probabilities'].argmax(dim=-1)

        y_true.extend(y.cpu())
        y_pred.extend([round(i) for i in output["probabilities"].cpu().detach().numpy()])
        
    final_acc = accuracy_score(y_true, y_pred)
    return { "accuracy" : final_acc }

@torch.no_grad()
def evaluate_accuracy_rnn(model: Module, dataloader: DataLoader):#
    y_true = []
    y_pred = []
    losses = []
    for x, x_len, y in dataloader:
        output = model(x, x_len, y)
        predictions = output['probabilities'].argmax(dim=-1)
        loss = output["loss"]
        losses.append(loss)

        y_true.extend(y)
        y_pred.extend([round(i) for i in output["probabilities"].detach().numpy()])
        
    final_acc = accuracy_score(y_true, y_pred)
    return { "accuracy" : final_acc, "loss" : np.mean(np.array(losses, dtype=np.float32))}

def train_evaluate(
        model: Module, 
        optimizer: Optimizer, 
        train_dataloader: DataLoader, 
        valid_dataloader: DataLoader, 
        valid_fn=None, 
        epochs: int=5, 
        verbose: bool=True,
        early_stopping: bool=True,
        patience: int=5,
        rnn: bool=False, 
        device: str="cpu"
    ):
    """
    Defines the training-validation loop with the given classifier and 
    the train and validation dataloaders.
    """
    valid_fn = evaluate_accuracy if not rnn else evaluate_accuracy_rnn
    patience_cnt = patience
    loss_history  = []
    train_history = []
    valid_history = []

    for epoch in range(epochs):
        losses = []
        y_true = []
        y_pred = []

        # batches of the training set
        for sample in train_dataloader:
            if len(sample) == 3:
                x = sample[0]
                x_len = sample[1]
                y = sample[2]
            else:
                x = sample[0].to(device)
                y = sample[1].to(device)

            optimizer.zero_grad()
            batch_out = model(x, y) if not rnn else model(x, x_len, y) 
            loss = batch_out["loss"]
            losses.append(loss)           
            loss.backward()     # computes the gradient of the loss
            optimizer.step()    # updates parameters based on the gradient information

            # to compute accuracy
            y_true.extend(y.cpu())
            y_pred.extend([round(i) for i in batch_out["probabilities"].cpu().detach().numpy()])

        model.global_epoch += 1
        mean_loss = sum(losses) / len(losses)
        loss_history.append(mean_loss.item())

        acc = accuracy_score(y_true, y_pred)
        train_history.append(acc)
        if verbose or epoch == epochs - 1:
            print(f'  Epoch {model.global_epoch:3d}/{epochs} => Loss: {mean_loss:0.6f}, \ttrain accuracy: {acc:0.4f}')
           
        if verbose and valid_dataloader:
            assert valid_fn is not None
            valid_output = valid_fn(model, valid_dataloader)
            valid_value = valid_output["accuracy"]
            valid_history.append(valid_value)
            valid_loss = valid_output["loss"]
            print(f"\t\tValidation => loss: {valid_loss:0.6f}, \taccuracy: {valid_value:0.6f}\n")
            if early_stopping:
                if patience_cnt <= 0:
                    print(f"[INFO]: Early stop! -> patience: {patience_cnt}")
                    break

                if epoch > 20 and valid_history[-1] >= valid_history[-2] and loss_history[-1] <= loss_history[-2]:
                    patience_cnt += 1
                    print(f"(patience inc: {patience_cnt})")

                elif epoch > 0 and valid_history[-1] < valid_history[-2]:
                    patience_cnt -= 1
                    print(f"(patience dec: {patience_cnt})")


    print("\n[INFO]: training summary...")
    print("[INFO]: accuracy scores...")
    print(f"avg_train: {np.mean(np.array(train_history)):0.6f}")
    print(f"avg_eval : {np.mean(np.array(valid_history)):0.6f}")
    print(f"max_train: {np.max(np.array(train_history)):0.6f}")
    print(f"max_eval : {np.max(np.array(valid_history)):0.6f}")

    history = {}
    history["train_loss"] = loss_history
    history["train_acc"] = train_history
    history["eval_acc"] = valid_history
    return history 


####### WordEmb classifier
class BaseMLPClassifier(Module):
    """ TODO
    This module defines a small MLP classifier
    """
    def __init__(self, input_features: int=50, hidden_size: int=100, output_classes: int=1):
        super().__init__()
        #self.emb_to_aggregation_layer = EmbAggregation(pretrained_emb)
        #self.input_feature = input_features*2 if self.emb_to_aggregation_layer.aggr_type == "concat" else input_features
        self.hidden1_layer = Linear(input_features, hidden_size)
        self.hidden2_layer = Linear(hidden_size, hidden_size)
        self.output_layer = Linear(hidden_size, output_classes)
        self.loss_fn = BCELoss()
        self.global_epoch = 0

    def forward(self, x: Tensor, y: Tensor=None):
        #aggregated_embs = self.emb_to_aggregation_layer(x) 
        hidden1 = self.hidden1_layer(x)
        hidden1_out = relu(hidden1)
        hidden2 = self.hidden2_layer(hidden1_out)
        hidden_output = relu(hidden2)
        
        logits = self.output_layer(hidden_output).squeeze(1)
        probabilities = sigmoid(logits)
        result = {'logits': logits, 'probabilities': probabilities}

        if y is not None:
            loss = self.loss(probabilities, y.float())
            result['loss'] = loss
        return result

    def loss(self, pred, y):
        return self.loss_fn(pred, y)


####### RNN classifier
def rnn_collate_fn(data_elements: list, pad=18000):  # data_elements is a list of (x, y) pairs
    """
    Override the collate function in order to deal with the different sizes of the input 
    index sequences. (data_elements is a list of ((x1, x2), y) tuples)
    """
    X_1 = []
    X_2 = []
    x1_lens = []
    x2_lens = []
    y = []
    for elem in data_elements:
        #print(elem[0])
        x1 = elem[0][0]
        x2 = elem[0][1]

        X_1.append(x1) # list of index tensors
        X_2.append(x2)
        x1_lens.append(x1.size(0)) # to implement the many-to-one strategy
        x2_lens.append(x2.size(0)) # to implement the many-to-one strategy
        y.append(elem[1])

    X_1 = torch.nn.utils.rnn.pad_sequence(X_1, batch_first=True, padding_value=pad)
    X_2 = torch.nn.utils.rnn.pad_sequence(X_2, batch_first=True, padding_value=pad)
    x1_lens = torch.LongTensor(x1_lens)
    x2_lens = torch.LongTensor(x2_lens)
    y = Tensor(y)

    return (X_1,X_2), (x1_lens,x2_lens), y


class RecurrentLSTMClassifier(Module):
    """ TODO
    This module defines an RNN embeddings aggregation step followed by a small MLP classifier.
    """
    def __init__(self, pretrained_emb=None, hidden_size: int=128, output_classes: int=1, aggr_type: str="sub"):
        super().__init__()
        self.aggr_type = "sub"
        self.get_last = False
        self.global_epoch = 0

        # embedding layer
        self.embedding = embedding_lookUp(pretrained_emb)

        # recurrent layers: one lstm for each sentence of a couple
        self.rnn1 = LSTM(
            input_size=pretrained_emb.shape[1], 
            hidden_size=hidden_size//2, 
            num_layers=1, 
            batch_first=True,
            bidirectional=True
        )
        #self.rnn2 = LSTM(
        #    input_size=pretrained_emb.shape[1],
        #    hidden_size=hidden_size//4, 
        #    num_layers=1, 
        #    batch_first=True,
        #    bidirectional=True
        #)
        
        # linear layers 
        self.hidden1_layer = Linear(hidden_size, hidden_size)
        self.hidden2_layer = Linear(hidden_size, hidden_size//2)
        self.output_layer = Linear(hidden_size//2, output_classes)
        self.loss_fn = BCELoss()

    def forward(self, x: Tensor, x_len: Tensor, y: Tensor=None):
        # embedding words from indices
        #embedding_out = self.embedding(x.long())
        emb1_out = self.embedding(x[0].long())
        emb2_out = self.embedding(x[1].long())

        lats1_idx = x_len[0] - 1
        lats2_idx = x_len[1] - 1

        # recurrent encoding -> rnn1
        batch_s1 = []
        batch_s2 = []

        rnn1_out = self.rnn1(emb1_out)[0]
        rnn2_out = self.rnn1(emb2_out)[0]
        batch_size, _, hidden_size = rnn1_out.shape
        for i in range(batch_size):
            s1_words_rnn = rnn1_out[i][:lats1_idx[i]]
            s2_words_rnn = rnn2_out[i][:lats2_idx[i]]

            #print("batch elem:", batch_elem.size())
            s1_words_avg = torch.mean(s1_words_rnn, dim=0)
            s2_words_avg = torch.mean(s2_words_rnn, dim=0)
            batch_s1.append(s1_words_avg)
            batch_s2.append(s2_words_avg)

        avg_s1 = torch.stack(batch_s1)
        avg_s2 = torch.stack(batch_s2)
        #print("avg s1 rnn out:", avg_s1.size())
        #print("avg s2 rnn out:", avg_s2.size())
        vectors_1 = avg_s1
        vectors_2 = avg_s2

        if self.get_last:
            rnn1_out = self.rnn1(emb1_out)[0]
            batch_size, seq1_len, hidden_size = rnn1_out.shape
            flat1_out = rnn1_out.reshape(-1, hidden_size)

            pads_seq1 = torch.arange(batch_size) * seq1_len
            vec1_idxs = pads_seq1 + lats1_idx

            # recurrent encoding -> rnn2
            rnn2_out = self.rnn1(emb2_out)[0]
            seq2_len = rnn2_out.shape[1]
            flat2_out = rnn2_out.reshape(-1, hidden_size)
            
            pads_seq2 = torch.arange(batch_size) * seq2_len
            vec2_idxs = pads_seq2 + lats2_idx

            vectors_1_end = flat1_out[pads_seq1 + 0]
            vectors_2_end = flat2_out[pads_seq2 + 0]
            vectors_1 = flat1_out[vec1_idxs]
            vectors_2 = flat2_out[vec2_idxs]

        if VERBOSE:
            #print("\npads:", pads_seq1)
            print("\nlats:", lats1_idx)
            print("rnn1 out shape:", rnn1_out.size())
            #print("vec1_idxs:", vec1_idxs.size())
            #print("Flat out:", flat1_out.size())


        if self.aggr_type == "cat":
            vec_summary = torch.cat([vectors_1,vectors_2], dim=1).float()
        elif self.aggr_type == "sub":
            vec_summary = torch.abs(torch.sub(vectors_1, vectors_2)).float()
        elif self.aggr_type == "sqr_sub":
            s1_sqr = torch.mul(vectors_1,vectors_1)
            s2_sqr = torch.mul(vectors_2,vectors_2)
            vec_summary = torch.abs(torch.sub(s1_sqr, s2_sqr)).float()

        if VERBOSE:
            print("Vec1 shape:", vectors_1.size())
            print("vecsum shape:", vec_summary.size(), "\n")

        hidden1 = self.hidden1_layer(vec_summary)
        hidden1_out = relu(hidden1)
        hidden2 = self.hidden2_layer(hidden1_out)
        hidden_output = relu(hidden2)

        logits = self.output_layer(hidden_output).squeeze(1)
        preds = sigmoid(logits)
        result = {'logits': logits, 'probabilities': preds}

        # compute loss
        if y is not None:
            loss = self.loss(preds, y.float())
            result['loss'] = loss

        return result

    def loss(self, pred, y):
        return self.loss_fn(pred, y)