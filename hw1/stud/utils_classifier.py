from torch import Tensor, load
from torch import relu, sigmoid
from torch.nn import Module, Linear, BCELoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score

def load_saved_model(save_path: str, input_dim: int = 200):
    """
    Loads a saved and pre-trained model, generated with the FooClassifier class.
    """
    print(f"\n[INFO]: loading pre-trained model from '{save_path}'...")
    model = FooClassifier(input_features=input_dim)
    model.load_state_dict(load(save_path))
    model.eval()
    print(model.state_dict().keys())
    return model

def train_loop(model: Module, optimizer: Optimizer, train_dataloader: DataLoader, 
            epochs: int = 5, verbose: bool = True):
    """
    Defines the training loop with the given classifier module.
    """
    loss_history = []
    acc_history  = []

    for epoch in range(epochs):
        losses = []
        y_true = []
        y_pred = []

        # batches of the training set
        for x, y in train_dataloader:
            optimizer.zero_grad()
            batch_out = model(x.cuda(), y.cuda())
            loss = batch_out["loss"]
            losses.append(loss)           
            loss.backward()     # computes the gradient of the loss
            optimizer.step()    # updates parameters based on the gradient information

            # to compute accuracy
            y_true.extend(y)
            y_pred.extend([round(i) for i in batch_out["probabilities"].cpu().detach().numpy()])

        model.global_epoch += 1
        mean_loss = sum(losses) / len(losses)
        loss_history.append(mean_loss.item())

        acc = accuracy_score(y_true, y_pred)
        acc_history.append(acc)
        if verbose or epoch == epochs - 1:
            print(f'  Epoch {model.global_epoch:3d} => Loss: {mean_loss:0.6f}, \ttrain accuracy: {acc:0.4f}')
            print('    ---------------')
    
    return {"loss": loss_history, "accuracy": acc_history}

def evaluate_model(model: Module, test_dataloader: DataLoader):
    y_true = []
    y_pred = []
    acc_history = []
    loss_history = []

    for x, y in test_dataloader:
        batch_true = []
        batch_pred = []
        output = model(x, y)
        loss = output["loss"]
        preds = [round(i) for i in output["probabilities"].detach().numpy()]

        # compute batch loss
        loss_history.append(loss)

        # compute batch accuracy
        batch_true.extend(y)
        batch_pred.extend(preds)

        batch_acc = accuracy_score(batch_true, batch_pred)
        acc_history.append(batch_acc)
        print("\tbatch val accuracy:", batch_acc)

        # for overall accuracy
        y_true.extend(y)
        y_pred.extend([round(i) for i in output["probabilities"].detach().numpy()])
    
    final_acc = accuracy_score(y_true, y_pred)
    print("Final val accuracy:", final_acc)
    return {"loss": loss_history, "accuracy": acc_history}


class FooClassifier(Module):
    """ TODO
    This module defines a small MLP classifier
    """
    def __init__(self, input_features: int, hidden_size: int = 64, output_classes: int = 1):
        super().__init__()
        self.hidden_layer = Linear(input_features, hidden_size).cuda()
        self.output_layer = Linear(hidden_size, output_classes).cuda()
        self.loss_fn = BCELoss().cuda()
        self.global_epoch = 0
        #self.cuda(device)

    def forward(self, x: Tensor, y: Tensor):
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