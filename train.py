import torch.nn
import time
import tqdm
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR


# Function for training a model with the given optimizer, to make the subsequent code more readable and modular
def train_chosen_optimizer(model, optimizer: Optimizer, training_loader, test_loader, device, total_epochs):
    loss_function = torch.nn.CrossEntropyLoss()       # Standard loss for classification
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)     # Scheduler to decrease learning rate during training
    losses = []
    steps = []
    times = []
    step = 0

    # Measure the start time of the training
    start_time = time.time()
    for current_epoch in range(total_epochs):
        print(f"Epoch: {current_epoch + 1}/{total_epochs}\n")

        for batch, (feature_vector, label) in enumerate(tqdm.tqdm(training_loader)):    # To show the progress bar
            data = feature_vector.to(device)
            targets = label.to(device)

            # Feedforward
            scores = model(data)
            loss = loss_function(scores, targets)
            losses.append(loss.item())
            steps.append(len(losses))

            # Backprogpagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(lambda: loss)    # Provide the current loss to the optimizer (needed for the loss_dependent_frequency schedule)

            step += 1

            # Measure the time after the current iteration
            times.append(time.time() - start_time)

        # After each epoch, change the learning rate according to the scheduler
        scheduler.step()

    return (steps, losses, times)
