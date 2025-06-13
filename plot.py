import matplotlib.pyplot as plt
import os

# Plots all relevant plots (Loss per iteration, 
# loss per time step, average loss per epoch and average learning dynamic over time)
def plot(stepsLosses, timesLosses, num_epochs):
    dirname = 'plots/'
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, dirname)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    plt.figure()
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.grid(True)

    for name, (losses, steps) in stepsLosses:
        plt.plot(steps, losses, label=name)

    plt.legend()
    plt.savefig(results_dir + 'loss_over_iterations')
    
    plt.figure()
    plt.xlabel("Time")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.grid(True)

    for name, (losses, times) in timesLosses:
        plt.plot(times, losses, label=name)

    plt.legend()
    plt.savefig(results_dir + 'loss_over_time')
    
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.grid(True)
    for name, (losses, steps) in stepsLosses:
        plt.plot([i + 1 for i in range(num_epochs)], [sum([losses[i] for i in range(250 * epoch, 250 * (epoch + 1))])/250 for epoch in range(num_epochs)], label=name)

    plt.legend()
    plt.savefig(results_dir + 'average_loss_over_epoch')
    
    plt.figure()
    plt.xlabel("Time")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.grid(True)
    for name, (losses, times) in timesLosses:
        plt.plot([times[i] for i in range(len(times)) if i % 125 == 0] , [losses[i] for i in range(len(losses)) if i % 125 == 0], label=name)

    plt.legend()
    plt.savefig(results_dir + 'averag_learning_dynamics_over_epoch')