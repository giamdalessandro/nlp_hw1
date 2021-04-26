"""
def train_loop(model: Module, optimizer: Optimizer, train_dataloader: DataLoader, 
            epochs: int = 5, verbose: bool = True, device: str="cpu"):
    loss_history = []
    acc_history  = []

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



def eval_saved_model():
    ## create evaluation Dataset instance
    dev_dataset = WordEmbDataset(DEV_PATH, DEV_VOCAB_SIZE, UNK, SEP, merge=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)

    # load the saved model
    dev_model = load_saved_model(os.path.join(SAVE_PATH, "train110_glove100d.pt"))

    # evaluate the model with the dev data
    print("\n[INFO]: Beginning evaluation ...\n")
    #history = evaluate_model(dev_model, dev_dataloader)

"""