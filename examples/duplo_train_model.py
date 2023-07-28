from tqdm import trange, tqdm

import argparse
import pickle as pkl

import numpy as np
import torch
import torch.nn.functional as F
import sys

from frankapy.nn import NeuralNet

import matplotlib.pyplot as plt

import os

import time


def evaluate_epoch(model, X_train, Y_train, X_val, Y_val, l1_gain, e, print_every):
    model.eval()

    with torch.no_grad():
        # There might be inconsistencies between total loss and batch-loss, but probably is fine
        #  because l1-reg is sum and mse_loss is mean
        l1_loss = 0.
        for param in model.parameters():
            l1_loss += F.l1_loss(param, torch.zeros_like(param), reduction='sum')
        l1_loss = l1_gain * l1_loss
        fc_hat, Kp = model(X_train)
        assert fc_hat.shape == Y_train.shape, f"fc_hat.shape = {fc_hat.shape}, Y_train.shape = {Y_train.shape}"
        total_train_loss = F.mse_loss(fc_hat, Y_train) + l1_loss
        print(f'The total training loss for epoch {e + 1} is: {total_train_loss.item()} '
              f'(mse_loss = {total_train_loss.item() - l1_loss}, l1_loss = {l1_loss})')
        fc_hat_val, Kp_val = model(X_val)
        assert fc_hat_val.shape == Y_val.shape, f"fc_hat_val.shape = {fc_hat_val.shape}, Y_val.shape = {Y_val.shape}"
        total_val_loss = F.mse_loss(fc_hat_val, Y_val) + l1_loss
        print(f'The total validation loss for epoch {e + 1} is: {total_val_loss.item()} '
              f'(mse_loss = {total_val_loss.item() - l1_loss}, l1_loss = {l1_loss})')

        '''if (e + 1) % print_every == 0:
            demo_trans_stiffness = data_dict['experimental_setup']['demo_trans_stiffness']
            plt.figure(figsize=(8.0, 5.0))
            plt.subplot(311)
            for j in range(n_demos - int(use_validation)):
                plt.plot(ts, Kp[j * N:(j + 1) * N, 0].cpu())
            plt.plot([ts[0], ts[-1]], [demo_trans_stiffness[0], demo_trans_stiffness[0]], 'r--')
            # plt.hlines([K_min, K_max], ts[0], ts[-1], 'b', '--')
            plt.grid()
            plt.ylim(0, K_max)
            plt.title('Stiffness in x-direction')
            plt.ylabel(r'$k_x$')
            plt.subplot(312)
            for j in range(n_demos - int(use_validation)):
                plt.plot(ts, Kp[j * N:(j + 1) * N, 1].cpu())
            plt.plot([ts[0], ts[-1]], [demo_trans_stiffness[1], demo_trans_stiffness[1]], 'r--')
            # plt.hlines([K_min, K_max], ts[0], ts[-1], 'b', '--')
            plt.grid()
            plt.ylim(0, K_max)
            plt.title('Stiffness in y-direction')
            plt.ylabel(r'$k_y$')
            plt.subplot(313)
            for j in range(n_demos - int(use_validation)):
                plt.plot(ts, Kp[j * N:(j + 1) * N, 2].cpu())
            plt.plot([ts[0], ts[-1]], [demo_trans_stiffness[2], demo_trans_stiffness[2]], 'r--')
            # plt.hlines([K_min, K_max], ts[0], ts[-1], 'b', '--')
            plt.grid()
            plt.ylim(0, K_max)
            plt.title('Stiffness in z-direction')
            plt.ylabel(r'$k_z$')
            plt.xlabel('Time [s]')
            plt.suptitle(f'Estimated stiffnesses for demonstration trajectories after training for {e + 1} epochs')
            plt.tight_layout()
            plt.show()

            if use_validation:
                plt.figure(figsize=(8.0, 5.0))
                plt.subplot(311)
                plt.plot(ts, Kp_val[:, 0].cpu())
                plt.plot([ts[0], ts[-1]], [demo_trans_stiffness[0], demo_trans_stiffness[0]], 'r--')
                # plt.hlines([K_min, K_max], ts[0], ts[-1], 'b', '--')
                plt.grid()
                plt.ylim(0, K_max)
                plt.title('Stiffness in x-direction')
                plt.ylabel(r'$k_x$')
                plt.subplot(312)
                plt.plot(ts, Kp_val[:, 1].cpu())
                plt.plot([ts[0], ts[-1]], [demo_trans_stiffness[1], demo_trans_stiffness[1]], 'r--')
                # plt.hlines([K_min, K_max], ts[0], ts[-1], 'b', '--')
                plt.grid()
                plt.ylim(0, K_max)
                plt.title('Stiffness in y-direction')
                plt.ylabel(r'$k_y$')
                plt.subplot(313)
                plt.plot(ts, Kp_val[:, 2].cpu())
                plt.plot([ts[0], ts[-1]], [demo_trans_stiffness[2], demo_trans_stiffness[2]], 'r--')
                # plt.hlines([K_min, K_max], ts[0], ts[-1], 'b', '--')
                plt.grid()
                plt.ylim(0, K_max)
                plt.title('Stiffness in z-direction')
                plt.ylabel(r'$k_z$')
                plt.xlabel('Time [s]')
                plt.suptitle(f'Estimated stiffnesses for validation trajectory after training for {e + 1} epochs')
                plt.tight_layout()
                plt.show()'''
    return [total_train_loss.item(), total_val_loss.item(), l1_loss.item()]


def train_nn(model, optimizer, X_train, Y_train, X_val, Y_val, batch_size, epochs=1, l1_gain=0., print_every=25):
    """
    Description of function
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model.
    - X_train: (N*n_demos, 30) tensor: training data fed to the network
    - Y_train: (N*n_demos, 6) tensor: target data
    - batch_size
    - device: device used for training (CPU/GPU)
    - dtype
    - epochs: The number of epochs to train for
    - l1_gain: L1 regularizing factor

    Returns: nothing
    """
    # model = model.to(device=device)

    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    l1_losses = np.zeros(epochs)
    for e in range(epochs):
        model.train()
        # indices = torch.arange(X_train.shape[0])
        indices = torch.randperm(X_train.shape[0])  # random permutations of the indices
        sys.stdout.flush()
        with tqdm(total=X_train.shape[0], desc=f'Training') as pbar:
            for start_idx in range(0, X_train.shape[0], batch_size):
                last_idx = min(start_idx + batch_size, X_train.shape[0])
                batch_idx = indices[start_idx:last_idx]

                inputs = X_train[batch_idx]
                truths = Y_train[batch_idx]

                optimizer.zero_grad(set_to_none=True)

                inputs.required_grad = True
                output, _ = model(inputs)
                loss = F.mse_loss(output, truths)

                # Add L1 regularisation
                l1_loss = 0.
                for param in model.parameters():
                    l1_loss += F.l1_loss(param, torch.zeros_like(param), reduction='sum')
                loss += l1_gain * l1_loss

                loss.backward()
                optimizer.step()

                step_metrics = {'loss': loss.item()}
                pbar.set_postfix(**step_metrics)
                pbar.update(list(inputs.shape)[0])
        sys.stdout.flush()

        train_loss_e, val_loss_e, l1_loss_e = evaluate_epoch(model, X_train, Y_train, X_val, Y_val, l1_gain, e,
                                                             print_every)
        train_losses[e] = train_loss_e
        val_losses[e] = val_loss_e
        l1_losses[e] = l1_loss_e

    # Save loss metrics progression per epoch (train_loss, val_loss, l1_loss)
    model_dict['train_losses'] = train_losses
    model_dict['val_losses'] = val_losses
    model_dict['l1_losses'] = l1_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_file', '-rf', default='data_dict')
    parser.add_argument('--save_file', '-sf', default='model_dict')
    parser.add_argument('--dir', '-d', default='')
    args = parser.parse_args()

    save_model = False
    use_validation = True
    USE_GPU = False

    # Hyperparameters:
    n_epochs = 100
    lr = 1e-4
    K_min = 0.1
    K_max = 1500
    hidden_layers = [32, 16]  # Use [32, 16] (baseline) and [256, 128, 64]
    # hidden_layers_arr = [[32, 16],
    #                     [256, 128, 64]]
    batch_size = 64
    l1_gain = 5e-4  # Use [5e-4, 5e-5]
    # l1_gain_arr = [5e-4, 5e-5]

    assert os.path.isdir(args.dir) or args.dir == '', "Folder does not exist - cannot load data"

    # Read parameters from read-file: rotational_stiffness, X, Y, dt
    with open(args.dir + '/' + args.read_file + '.pkl', 'rb') as pkl_f:
        data_dict = pkl.load(pkl_f)
        print(f'Did load data dict: {args.read_file}')

    # Load data parameters
    rotational_stiffness = data_dict['experimental_setup']['rotational_stiffness']
    dt = data_dict['experimental_setup']['dt']
    n_demos = len(data_dict['demo'])
    duration = data_dict['experimental_setup']['duration']
    ts = np.arange(0, duration, dt)
    N = len(ts)

    assert n_demos >= 1 + int(use_validation)

    print('Initialising Pytorch')
    dtype = torch.float32
    device = torch.device('cuda') if USE_GPU and torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)

    #for i_model, hidden_layers in enumerate(hidden_layers_arr):

    model_name = str(hidden_layers[0])
    for k in range(1, len(hidden_layers)):
        model_name += '_' + str(hidden_layers[k])
    model_dict = {}
    print(f'Training model {model_name} with data from {n_demos - 1} demonstrations.\n')
    # l1_gain = l1_gain_arr[i_model]

    # Collect NN data input (X) and true labels (Y)
    for j in range(n_demos):
        positions_j = data_dict['demo'][j]['positions']
        quaternions_j = data_dict['demo'][j]['quaternions']
        xe_j = np.concatenate((positions_j, quaternions_j[:, 1:]), axis=1)
        ve_j = data_dict['demo'][j]['ve']
        he_hat_j = data_dict['demo'][j]['he_hat']
        target_positions_j = data_dict['demo'][j]['target_positions']
        target_quaternions_j = data_dict['demo'][j]['target_quaternions']
        xd_j = np.concatenate((target_positions_j, target_quaternions_j[:, 1:]), axis=1)
        vd_j = data_dict['demo'][j]['vd']
        hc_j = data_dict['demo'][j]['hc']

        X_j = torch.tensor(np.concatenate((xe_j, ve_j, he_hat_j, xd_j, vd_j), axis=1), dtype=dtype, device=device)
        X = X_j if j == 0 else torch.cat((X, X_j), 0)
        Y_j = torch.tensor(hc_j[:, :3], dtype=dtype, device=device)
        Y = Y_j if j == 0 else torch.cat((Y, Y_j), 0)

    if use_validation:
        X_train, Y_train = X[:N * (n_demos - 1)], Y[:N * (n_demos - 1)]
        X_val, Y_val = X[(n_demos - 1) * N:], Y[(n_demos - 1) * N:]
    else:
        X_train, Y_train = X, Y
        X_val, Y_val = None, None

    # Initialise and train neural network for n_epochs
    model = NeuralNet(rotational_stiffness=torch.tensor(rotational_stiffness, dtype=dtype, device=device),
                      K_min=K_min, K_max=K_max, hidden_layers=hidden_layers).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    timer_start = time.time()
    train_nn(model, optimizer, X_train, Y_train, X_val, Y_val, batch_size, n_epochs, l1_gain,
             print_every=round(n_epochs / 1))
    timer_end = time.time()

    print(f'Total training time: {timer_end - timer_start} seconds')

    model_dict['n_epochs'] = n_epochs
    model_dict['lr'] = lr
    model_dict['K_min'] = K_min
    model_dict['K_max'] = K_max
    model_dict['hidden_layers'] = hidden_layers
    model_dict['batch_size'] = batch_size
    model_dict['l1_gain'] = l1_gain
    model_dict['use_validation'] = use_validation

    model.eval()
    with torch.no_grad():
        fc_hat, Kp_hat = model(X)
    for j in range(n_demos):
        fc_hat_j = fc_hat[j * N:(j + 1) * N].detach().cpu().numpy()
        Kp_hat_j = Kp_hat[j * N:(j + 1) * N].detach().cpu().numpy()

        model_dict_j = {'fc_hat': fc_hat_j, 'Kp_hat': Kp_hat_j}
        model_dict[j] = model_dict_j

    # Save model and model-dict to file (.pkl and .mat)
    if save_model:
        torch.save(model.state_dict(), f'{args.dir}/model_state_dict_{model_name}.pt')
        with open(f'{args.dir}/{args.save_file}_{model_name}.pkl', 'wb') as pkl_f:
            pkl.dump(model_dict, pkl_f)
            print(f'Did save model: {args.dir}/{args.save_file}_{model_name}.pkl')

    # Plot training progress
    plt.figure(figsize=(8.0, 5.0))
    plt.plot(np.arange(1, n_epochs + 1), model_dict['train_losses'])
    plt.plot(np.arange(1, n_epochs + 1), model_dict['val_losses'])
    plt.plot(np.arange(1, n_epochs + 1), model_dict['train_losses'] - model_dict['l1_losses'], '--')
    plt.plot(np.arange(1, n_epochs + 1), model_dict['val_losses'] - model_dict['l1_losses'], '--')
    plt.plot(np.arange(1, n_epochs + 1), model_dict['l1_losses'], '.-')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend([r'Total training loss (including $L_1$)', r'Total validation loss (including $L_1$)',
                'MSE training loss', 'MSE validation loss', r'Regularisation loss ($L_1$)'])
    plt.suptitle(f'Training progress per epoch for model: {model_name}')
    plt.tight_layout()
    # plt.savefig(f'{args.dir}/training_progress_{model_name}.eps', format='eps', bbox_inches='tight')
    plt.show()

    # Plot nominal trajectories
    plt.figure(figsize=(8.0, 5.0))
    plt.subplot(311)
    ax = plt.gca()
    for j in range(n_demos - int(use_validation)):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(data_dict['demo'][j]['times'], data_dict['demo'][j]['positions'][:, 0],
                 ts, data_dict['demo'][j]['target_positions'][:, 0], '--', color=color)
    plt.grid()
    plt.legend(['Actual', 'Desired'])
    plt.title('Trajectory x')
    plt.ylabel(r'$x$ [m]')
    plt.subplot(312)
    ax = plt.gca()
    for j in range(n_demos - int(use_validation)):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(data_dict['demo'][j]['times'], data_dict['demo'][j]['positions'][:, 1],
                 ts, data_dict['demo'][j]['target_positions'][:, 1], '--', color=color)
    plt.grid()
    plt.legend(['Actual', 'Desired'])
    plt.title('Trajectory y')
    plt.ylabel(r'$y$ [m]')
    plt.subplot(313)
    ax = plt.gca()
    for j in range(n_demos - int(use_validation)):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(data_dict['demo'][j]['times'], data_dict['demo'][j]['positions'][:, 2],
                 ts, data_dict['demo'][j]['target_positions'][:, 2], '--', color=color)
    plt.grid()
    plt.legend(['Actual', 'Desired'])
    plt.title('Trajectory z')
    plt.ylabel(r'$z$ [m]')
    plt.xlabel('Time [s]')
    plt.suptitle('Actual and desired demonstration trajectories')
    plt.tight_layout()
    # plt.savefig(args.dir + f'/position_demonstration.png',
    #            bbox_inches='tight', dpi=100)
    plt.show()
    if use_validation:
        plt.figure(figsize=(8.0, 5.0))
        plt.subplot(311)
        plt.plot(data_dict['demo'][n_demos - 1]['times'], data_dict['demo'][n_demos - 1]['positions'][:, 0],
                 ts, data_dict['demo'][n_demos - 1]['target_positions'][:, 0], '--')
        plt.grid()
        plt.legend(['Actual', 'Desired'])
        plt.title('Trajectory x')
        plt.ylabel(r'$x$ [m]')
        plt.subplot(312)
        plt.plot(data_dict['demo'][n_demos - 1]['times'], data_dict['demo'][n_demos - 1]['positions'][:, 1],
                 ts, data_dict['demo'][n_demos - 1]['target_positions'][:, 1], '--')
        plt.grid()
        plt.legend(['Actual', 'Desired'])
        plt.title('Trajectory y')
        plt.ylabel(r'$y$ [m]')
        plt.subplot(313)
        plt.plot(data_dict['demo'][n_demos - 1]['times'], data_dict['demo'][n_demos - 1]['positions'][:, 2],
                 ts, data_dict['demo'][n_demos - 1]['target_positions'][:, 2], '--')
        plt.grid()
        plt.legend(['Actual', 'Desired'])
        plt.title('Trajectory z')
        plt.ylabel(r'$z$ [m]')
        plt.xlabel('Time [s]')
        plt.suptitle('Actual and desired validation trajectory')
        plt.tight_layout()
        # plt.savefig(args.dir + f'/position_validation.png',
        #            bbox_inches='tight', dpi=100)
        plt.show()
