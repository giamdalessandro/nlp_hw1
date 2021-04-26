import torch
from torch import Tensor, LongTensor, load
from torch import relu, sigmoid
from torch.nn import Module, Linear, BCELoss, LSTM, Embedding
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score
from utils_aggregation import EmbAggregation, embedding_lookUp


def load_saved_model(save_path: str, input_dim: int=200):
    """
    Loads a saved and pre-trained model, generated with the FooClassifier class.
    """
    print(f"\n[INFO]: loading pre-trained model from '{save_path}'...")
    model = FooClassifier(input_features=input_dim)
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

        y_true.extend(y.cpu())
        y_pred.extend([round(i) for i in output["probabilities"].cpu().detach().numpy()])
        
    final_acc = accuracy_score(y_true, y_pred)
    return { "accuracy" : final_acc }

def train_evaluate(
        model: Module, 
        optimizer: Optimizer, 
        train_dataloader: DataLoader, 
        valid_dataloader: DataLoader, 
        valid_fn = evaluate_accuracy, 
        epochs: int = 5, 
        verbose: bool = True, 
        device: str="cpu"
    ):
    """
    Defines the training-validation loop with the given classifier and 
    the train and validation dataloaders.
    """
    loss_history = []
    train_history  = []
    valid_history = []

    for epoch in range(epochs):
        losses = []
        y_true = []
        y_pred = []

        # batches of the training set
        for x, y in train_dataloader:
            if device == "cuda":
                x = x.cuda()
                y = y.cuda()

            optimizer.zero_grad()
            batch_out = model(x, y)
            loss = batch_out["loss"]
            losses.append(loss)           
            loss.backward()     # computes the gradient of the loss
            optimizer.step()    # updates parameters based on the gradient information

            # to compute accuracy
            # TODO: we need to move the tensor back to CPU to compute accuracy via scikit-learn
            y_true.extend(y.cpu())
            y_pred.extend([round(i) for i in batch_out["probabilities"].cpu().detach().numpy()])

        model.global_epoch += 1
        mean_loss = sum(losses) / len(losses)
        loss_history.append(mean_loss.item())

        acc = accuracy_score(y_true, y_pred)
        train_history.append(acc)
        if verbose or epoch == epochs - 1:
            print(f'  Epoch {model.global_epoch:3d} => Loss: {mean_loss:0.6f}, \ttrain accuracy: {acc:0.4f}')
           
        if verbose and valid_dataloader:
            assert valid_fn is not None
            valid_output = valid_fn(model, valid_dataloader)
            valid_value = valid_output["accuracy"]
            valid_history.append(valid_value)
            print(f"    Validation => accuracy: {valid_value:0.6f}\n")

    history = {}
    history["train_loss"] = loss_history
    history["train_acc"] = train_history
    history["eval_acc"] = valid_history
    return history 

def train_evaluate_rnn(
        model: Module, 
        optimizer: Optimizer, 
        train_dataloader: DataLoader, 
        valid_dataloader: DataLoader, 
        valid_fn = evaluate_accuracy_rnn, 
        epochs: int = 5, 
        verbose: bool = True, 
        device: str="cpu"
    ):
    """
    Defines the training-validation loop with the given classifier and 
    the train and validation dataloaders.
    """
    loss_history = []
    train_history  = []
    valid_history = []

    for epoch in range(epochs):
        losses = []
        y_true = []
        y_pred = []

        # batches of the training set
        for x, x_len, y in train_dataloader:
            if device == "cuda":
                x = x.cuda()
                x_len = x_len.cuda()
                y = y.cuda()

            optimizer.zero_grad()
            batch_out = model(x, x_len, y)
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
            print(f'  Epoch {model.global_epoch:3d} => Loss: {mean_loss:0.6f}, \ttrain accuracy: {acc:0.4f}')
           
        if verbose and valid_dataloader:
            assert valid_fn is not None
            valid_output = valid_fn(model, valid_dataloader)
            valid_value = valid_output["accuracy"]
            valid_history.append(valid_value)
            print(f"    Validation => accuracy: {valid_value:0.6f}\n")

    history = {}
    history["train_loss"] = loss_history
    history["train_acc"] = train_history
    history["eval_acc"] = valid_history
    return history 


####### WordEmb classifier
class FooClassifier(Module):
    """ TODO
    This module defines a small MLP classifier
    """
    def __init__(self, input_features: int, hidden_size: int=64, output_classes: int=1):
        super().__init__()
        #self.emb_to_aggregation_layer = EmbAggregation(pretrained_emb)
        #self.input_feature = input_features*2 if self.emb_to_aggregation_layer.aggr_type == "concat" else input_features
        self.hidden_layer = Linear(input_features, hidden_size)
        self.output_layer = Linear(hidden_size, output_classes)
        self.loss_fn = BCELoss()
        self.global_epoch = 0

    def forward(self, x: Tensor, y: Tensor=None):
        #aggregated_embs = self.emb_to_aggregation_layer(x) 
        hidden_output = self.hidden_layer(x)
        hidden_output = relu(hidden_output)
        
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
    X = []
    y = []
    x_lens = []
    for elem in data_elements:
        X.append(elem[0]) # list of index tensors
        y.append(elem[1])
        x_lens.append(elem[0].size(0)) # to implement the many-to-one strategy

    X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=0)
    y = Tensor(y)
    x_lens = torch.LongTensor(x_lens)
    return X, x_lens, y


class FooRecurrentClassifier(Module):
    """ TODO
    This module defines an RNN embeddings aggregation step followed by a small MLP classifier.
    """
    def __init__(self, pretrained_emb, hidden_size: int=128, output_classes: int=1):
        super().__init__()

        # embedding layer
        self.embedding = embedding_lookUp(pretrained_emb)

        # recurrent layers
        self.rnn = LSTM(input_size=pretrained_emb.shape[1], hidden_size=hidden_size, num_layers=1, batch_first=True)
        
        # linear layers
        self.hidden_layer = Linear(hidden_size, hidden_size).cuda()
        self.output_layer = Linear(hidden_size, output_classes).cuda()
        self.loss_fn = BCELoss().cuda()

        self.global_epoch = 0

    def forward(self, x: Tensor, x_len: Tensor, y: Tensor=None):
        # embedding words from indices
        embedding_out = self.embedding(x.long())

        # recurrent encoding
        recurrent_out = self.rnn(embedding_out)[0]
        batch_size, seq_len, hidden_size = recurrent_out.shape

        flattened_out = recurrent_out.reshape(-1, hidden_size)
        
        last_word_relative_indices = x_len - 1
        sequences_offsets = torch.arange(batch_size) * seq_len
        summary_vectors_indices = sequences_offsets + last_word_relative_indices
        summary_vectors = flattened_out[summary_vectors_indices]

        out = self.hidden_layer(summary_vectors.cuda())
        out = relu(out)
        out = self.output_layer(out).squeeze(1)

        logits = out
        preds = sigmoid(logits.cuda())
        result = {'logits': logits, 'probabilities': preds.cpu()}

        # compute loss
        if y is not None:
            loss = self.loss(preds, y.cuda().float())
            result['loss'] = loss

        return result

    def loss(self, pred, y):
        return self.loss_fn(pred, y)