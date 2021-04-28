import torch
import numpy as np

from torch import Tensor, LongTensor, load
from torch import relu, sigmoid
from torch.nn import Module, Linear, BCELoss, LSTM, Embedding
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score
from utils_aggregation import EmbAggregation, embedding_lookUp


def load_saved_model(save_path: str, input_dim: int=200):
    """
    Loads a saved and pre-trained model, generated with the BaseMLPClassifier class.
    """
    print(f"\n[INFO]: loading pre-trained model from '{save_path}'...")
    model = BaseMLPClassifier(input_features=input_dim)
    model.load_state_dict(load(save_path))
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
    for x, x_len, y in dataloader:
        output = model(x, x_len, y)
        predictions = output['probabilities'].argmax(dim=-1)

        y_true.extend(y)
        y_pred.extend([round(i) for i in output["probabilities"].detach().numpy()])
        
    final_acc = accuracy_score(y_true, y_pred)
    return { "accuracy" : final_acc }

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
            print(f"\t\tValidation => accuracy: {valid_value:0.6f}\n")
            if early_stopping:
                if patience_cnt <= 0:
                    print(f"[INFO]: Early stop! -> patience: {patience_cnt}")
                    break

                if epoch > 0 and valid_history[-1] < valid_history[-2]:
                    patience_cnt -= 1
                    print(f"(patience dec: {patience_cnt})")

                elif epoch > 20 and valid_history[-1] >= valid_history[-2] and loss_history[-1] <= loss_history[-2]:
                    patience_cnt += 1
                    print(f"(patience inc: {patience_cnt})")

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
    def __init__(self, input_features: int, hidden_size: int=100, output_classes: int=1):
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
def rnn_collate_fn(data_elements: list):  # data_elements is a list of (x, y) pairs
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
        x1 = elem[0][0]
        x2 = elem[0][1]

        X_1.append(x1) # list of index tensors
        X_2.append(x2)
        x1_lens.append(x1.size(0)) # to implement the many-to-one strategy
        x2_lens.append(x2.size(0)) # to implement the many-to-one strategy
        y.append(elem[1])

    X_1 = torch.nn.utils.rnn.pad_sequence(X_1, batch_first=True, padding_value=0)
    X_2 = torch.nn.utils.rnn.pad_sequence(X_2, batch_first=True, padding_value=0)
    x1_lens = torch.LongTensor(x1_lens)
    x2_lens = torch.LongTensor(x2_lens)
    y = Tensor(y)

    return (X_1,X_2), (x1_lens,x2_lens), y


class RecurrentLSTMClassifier(Module):
    """ TODO
    This module defines an RNN embeddings aggregation step followed by a small MLP classifier.
    """
    def __init__(self, pretrained_emb, hidden_size: int=64, output_classes: int=1):
        super().__init__()
        self.global_epoch = 0

        # embedding layer
        self.embedding = embedding_lookUp(pretrained_emb)

        # recurrent layers: one lstm for each sentence of a couple
        self.rnn1 = LSTM(
            input_size=pretrained_emb.shape[1], 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True
        )
        self.rnn2 = LSTM(
            input_size=pretrained_emb.shape[1],
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True
        )
        
        # linear layers 
        self.hidden1_layer = Linear(hidden_size, hidden_size)
        #self.hidden2_layer = Linear(hidden_size, hidden_size)
        self.output_layer = Linear(hidden_size, output_classes)
        self.loss_fn = BCELoss()

    def forward(self, x: Tensor, x_len: Tensor, y: Tensor=None):
        # embedding words from indices
        #embedding_out = self.embedding(x.long())
        emb1_out = self.embedding(x[0].long())
        emb2_out = self.embedding(x[1].long())

        # recurrent encoding -> rnn1
        rnn1_out = self.rnn1(emb1_out)[0]
        batch_size, seq1_len, hidden_size = rnn1_out.shape
        flat1_out = rnn1_out.reshape(-1, hidden_size)

        lats1_idx = x_len[0] - 1
        pads_seq1 = torch.arange(batch_size) * seq1_len
        vec1_idxs = pads_seq1 + lats1_idx

        # recurrent encoding -> rnn2
        rnn2_out = self.rnn2(emb2_out)[0]
        seq2_len = rnn2_out.shape[1]
        flat2_out = rnn2_out.reshape(-1, hidden_size)
        
        lats2_idx = x_len[1] - 1
        pads_seq2 = torch.arange(batch_size) * seq2_len
        vec2_idxs = pads_seq2 + lats2_idx
        
        vectors_1 = flat1_out[vec1_idxs]
        vectors_2 = flat2_out[vec2_idxs]
        #vec_summary = torch.cat([vectors_1,vectors_2], dim=1).float()
        vec_summary = torch.sub(vectors_1, vectors_2).float()

        hidden1 = self.hidden1_layer(vec_summary)
        #hidden1_out = relu(hidden1)
        #hidden2 = self.hidden2_layer(hidden1_out)
        hidden_output = relu(hidden1)

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