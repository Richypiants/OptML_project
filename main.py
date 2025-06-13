import torch
import numpy as np
import torchvision

from strategies import *
import data
from modified_soap import SOAP
import plot
import train 


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    optimizer_configs = [
    ("Adam", lambda model: torch.optim.Adam(model.parameters(), lr=0.005)),
    ("SOAP1", lambda model: SOAP(model.parameters(), lr=0.005, precondition_frequency=1)),
    ("SOAP10", lambda model: SOAP(model.parameters(), lr=0.005, precondition_frequency=10)),
    ("SOAP100", lambda model: SOAP(model.parameters(), lr=0.005, precondition_frequency=100)),
    ("SOAP300", lambda model: SOAP(model.parameters(), lr=0.005, precondition_frequency=300)),
    ("SOAPHalving", lambda model: SOAP(model.parameters(), lr=0.005, precondition_frequency=1024, precondition_frequency_routine=halving_frequency)),
    ("SOAPDoubling", lambda model: SOAP(model.parameters(), lr=0.005, precondition_frequency=1, precondition_frequency_routine=doubling_frequency)),
    ("SOAPFixedInterval", lambda model: SOAP(model.parameters(), lr=0.005, precondition_frequency=8, precondition_frequency_routine=fixed_interval_doubling_frequency)),
    ("SOAPLossChange", lambda model: SOAP(model.parameters(), lr=0.005, precondition_frequency=0.1, precondition_frequency_routine=loss_dependent_frequency)),
    ("SOAPDoublingClipped256", lambda model: SOAP(model.parameters(), lr=0.005, precondition_frequency=1, precondition_frequency_routine=doubling_frequency_clipped)),
    ]
    
    # Loading data set
    batch_size = 200
    training_loader, test_loader = data.loadData(batch_size)

    num_epochs = 10
    num_trainings = 5
    models = []
    
    for name, optimizer_function in optimizer_configs:
        trainings = []

        for i in range(num_trainings):
            # Instantiate a ResNet18 model
            model = torchvision.models.resnet18(weights=None)

            # Since the ResNet18 takes 224x224 images as inputs, change the input layer in order to avoid useless inputs which would behave like noise
            model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = torch.nn.Identity()

            model = model.to(device)
            optimizer = optimizer_function(model)

            trainings.append((model, optimizer))

        models.append((name, trainings))
    
    # Pairs (losses, avg_steps) or (losses, avg_times) for each type of optimizer
    stepsLosses = []
    timesLosses = []

    # Train all the models
    for name, trainings in models:
        losses_per_run = []
        times_per_run = []

        # For each model, train it num_trainings times and compute the average losses and times
        for model, optimizer in trainings:
            steps, losses, times = train.train_chosen_optimizer(model, optimizer, training_loader, 
                                                                test_loader, device, num_epochs)

            losses_per_run.append(losses)
            times_per_run.append(times)

        avg_losses = []
        avg_times = []

        # Average the loss dynamics and times for each of the identical models trained
        for step in range(len(steps)):
            step_losses = [losses[step] for losses in losses_per_run]
            step_times = [times[step] for times in times_per_run]

            avg_losses.append(sum(step_losses) / len(step_losses))
            avg_times.append(sum(step_times) / len(step_times))

        # This saves the data if one wants to use it later for plots
        try:
            np.savez(name + "PlotData.npz", steps=steps, losses=avg_losses, times=avg_times)
        except Exception as e:
            print(f"Error: {name} {e}")

        stepsLosses.append((name, (avg_losses, steps)))
        timesLosses.append((name, (avg_losses, avg_times)))
    
    # Plot result
    plot.plot(stepsLosses, timesLosses, num_epochs)

if __name__ == "__main__":
    main()