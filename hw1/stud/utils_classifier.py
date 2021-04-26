from torch import Tensor, load
from torch import relu, sigmoid
from torch.nn import Module, Linear, BCELoss, LSTM, Embedding
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score
from utils_aggregation import EmbAggregation


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

def eval_saved_model():
    ## create evaluation Dataset instance
    dev_dataset = WordEmbDataset(DEV_PATH, DEV_VOCAB_SIZE, UNK, SEP, merge=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)

    # load the saved model
    dev_model = load_saved_model(os.path.join(SAVE_PATH, "train110_glove100d.pt"))

    # evaluate the model with the dev data
    print("\n[INFO]: Beginning evaluation ...\n")
    history = evaluate_model(dev_model, dev_dataloader)

#@torch.no_grad()
def evaluate_accuracy(model: Module, dataloader: DataLoader):#
    #correct_predictions = 0
    #num_predictions = 0
    y_true = []
    y_pred = []
    for x, y in dataloader:
        output = model(x.cuda(), y.cuda())
        predictions = output['probabilities'].argmax(dim=-1)

        y_true.extend(y.cpu())
        y_pred.extend([round(i) for i in output["probabilities"].cpu().detach().numpy()])
        #correct_predictions += (predictions == y).sum()
        #num_predictions += predictions.shape[0]

    final_acc = accuracy_score(y_true, y_pred)
    #accuracy = correct_predictions / num_predictions
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

    def forward(self, x: Tensor, y: Tensor):
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


####### RNN Classifier
def rnn_collate_fn(data_elements):
    X, y, x_lens = []

    for de in data_elements:
        X.append(de[0]) # list of index tensors
        y.append(de[1])
        x_lens.append(X.size(0))  # to implement the many-to-one strategy

    X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=0)
    y = Tensor(y)
    x_lens = Tensor(x_lens, dtype=torch.long)
    return X, x_lens, y

"""
class FooRecurrentClassifier(Module):

    def __init__(self, pretrained_emb: list, input_features: int, hidden_size: int=64, output_classes: int=1):
        super().__init__()

        emb_dim = pretrained_emb.shape()
        self.embedding = Embedding.from_pretrained(pretrained_emb)

        self.rnn = LSTM(input_size=vectors_store.size(1), hidden_size=n_hidden, num_layers=1, batch_first=True)
        self.hidden_layer = Linear(input_features, hidden_size)
        self.output_layer = Linear(hidden_size, output_classes)
        self.loss_fn = BCELoss()
        self.global_epoch = 0

    def forward(self, x: Tensor, y: Tensor):
        # embedding words from indices
        embedding_out = self.embedding(X)

        # recurrent encoding
        recurrent_out = self.rnn(embedding_out)[0]
        batch_size, seq_len, hidden_size = recurrent_out.shape

        flattened_out = recurrent_out.reshape(-1, hidden_size)
        
        last_word_relative_indices = X_length - 1
        sequences_offsets = torch.arange(batch_size, device=self.device) * seq_len
        summary_vectors_indices = sequences_offsets + last_word_relative_indices
        summary_vectors = flattened_out[summary_vectors_indices]

        out = self.lin1(summary_vectors)
        out = torch.relu(out)
        out = self.lin2(out).squeeze(1)

        logits = out
        pred = torch.softmax(logits, dim=-1)
        result = {'logits': logits, 'pred': pred}

        # compute loss
        if y is not None:
            loss = self.loss(logits, y)
            result['loss'] = loss

        return result

    def loss(self, pred, y):
        return self.loss_fn(pred, y)
"""