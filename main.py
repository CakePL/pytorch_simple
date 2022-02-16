"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of hand-written digit recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

"""
import time
import collections
import os
from functools import partial

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

DEVICE = torch.device("cpu")
BATCHSIZE = 128
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 10
LOG_INTERVAL = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10


def define_model(trial):
    n_layers = 2
    layers = []

    in_features = 28 * 28
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


def get_mnist():
    # Load FashionMNIST dataset.
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(DIR, train=True, download=True, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(DIR, train=False, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )

    return train_loader, valid_loader


def objective(trial):

    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    # Get the FashionMNIST dataset.
    train_loader, valid_loader = get_mnist()

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


def make_dist(importance, p):
    keys = list(importance.keys())
    values = list(importance.values())

    s = 0
    for val in values:
        val = val**p
        s += val
    for val in values:
        val /= s

    for i in range(len(values)):
        next_val = 0
        if i + 1 < len(values):
            next_val = values[i + 1]
        values[i] = (i + 1) * (values[i] - next_val)
    return collections.OrderedDict(zip(keys, values))


def print_dict(d, prefix):
    for key, value in d.items():
        print(prefix + "{}: {}".format(key, value))


def print_trial(trial):
    print("  Value: ", trial.value)
    print("  Params: ")
    print_dict(trial.params, "    ")


def statistics(study):
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    trial = study.best_trial
    print(f"Best trial: {trial.number}")
    print_trial(trial)

    importance = optuna.importance.get_param_importances(study)
    print("Importance:")
    print_dict(importance, "  ")
    return trial.value


def my_optimize_epochs(objective, timeout, n_epochs=3, zero_epoch=None, plots=False, p=1):
    if zero_epoch is None:
        zero_epoch = 1 / n_epochs

    start_time = time.time()
    zero_epoch_time = timeout * zero_epoch
    epoch_time = (timeout - zero_epoch_time) / n_epochs
    study = optuna.create_study(direction="maximize")
    sampler = study.sampler

    print("\nEPOCH 0")
    study.optimize(objective, timeout=zero_epoch_time)
    trial_count = len(study.get_trials(deepcopy=False, states=[TrialState.COMPLETE]))
    if trial_count < 2:
        study.optimize(objective, n_trials=2 - trial_count)

    for i in range(1, n_epochs):
        importance = optuna.importance.get_param_importances(study)
        distribution = make_dist(importance, p)
        print(f"\nEPOCH {i}")
        print("Importance:")
        print_dict(importance, "  ")
        print("Distribution:")
        print_dict(distribution, " ")
        fixed_params = {}

        for key, val in reversed(distribution.items()):
            study.optimize(objective, timeout=epoch_time * val)
            fixed_params[key] = study.best_params[key]
            study.sampler = optuna.samplers.PartialFixedSampler(fixed_params, sampler)

    statistics(study)
    stop_time = time.time()
    duration = stop_time - start_time
    print(f"Time: {duration}")
    if plots:
        show_plots(study)
    return study.best_trial.value, duration


def show_plots(study):
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_intermediate_values(study).show()
    optuna.visualization.plot_parallel_coordinate(study).show()
    optuna.visualization.plot_contour(study).show()
    optuna.visualization.plot_slice(study).show()
    optuna.visualization.plot_param_importances(study).show()
    optuna.visualization.plot_edf(study).show()


def my_optimize_standard(objective, timeout=None, print_complete_trials=False, n_trials=None, plots=False):
    start_time = time.time()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, timeout=timeout, n_trials=n_trials)
    statistics(study)
    stop_time = time.time()
    duration = stop_time - start_time
    print(f"Time: {duration}")
    if print_complete_trials:
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        for trial in complete_trials:
            print(f"Trial {trial.number}:")
            print_trial(trial)
        fig = optuna.visualization.plot_slice(study)
        print(fig)
        fig.show()
    if plots:
        show_plots(study)
    return study.best_trial.value, duration, study


#its set to optimize only hhparameter p, but it can be used to optimize hhparameters n_epochs or zero_epoch
def meta_objective(trial, timeout, n_epochs):
    p = trial.suggest_float("p", 0.5, 2.0, log=True)
    N = 8
    return mean([my_optimize_epochs(objective, timeout=timeout, n_epochs=n_epochs, p=p)[0] for _ in range(N)])


def my_metaoptimize(n_trials, internal_timeout, internal_n_epochs=3):
    return my_optimize_standard(partial(meta_objective, timeout=internal_timeout, n_epochs=internal_n_epochs),
                                n_trials=n_trials,
                                timeout=None,
                                print_complete_trials=True)


def mean(li):
    s = 0
    for val in li:
        s += val
    return s / len(li)


if __name__ == "__main__":
    result_study = my_metaoptimize(n_trials=36, internal_timeout=300)[2]



