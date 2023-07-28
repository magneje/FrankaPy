# from tqdm import trange, tqdm

# import rospy
import argparse
import pickle as pkl
# import scipy.io
# import subprocess

# from bagpy import bagreader
# import pandas as pd

import numpy as np
# import torch
# import torch.nn.functional as F
# import sys

# from utilities import demonstrate_trajectory
# from frankapy.nn import NeuralNet
# from frankapy.utils import franka_pose_to_rigid_transform

# import quaternion

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import math

import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', '-df', default='data_dict')
    parser.add_argument('--model_file', '-mf', default='model_dict_32_16')
    parser.add_argument('--dir', '-d', default='')
    args = parser.parse_args()

    save_figures = True

    assert os.path.isdir(args.dir) or args.dir == '', "Folder does not exist - cannot load data"

    # LOAD DATA
    with open(args.dir + '/' + args.data_file + '.pkl', 'rb') as pkl_f:
        data_dict = pkl.load(pkl_f)
        print(f'Did load data dict: {args.data_file}')
    with open(args.dir + '/' + args.model_file + '.pkl', 'rb') as pkl_f:
        model_dict = pkl.load(pkl_f)
        print(f'Did load model dict: {args.model_file}')

    # Load data parameters
    rotational_stiffness = data_dict['experimental_setup']['rotational_stiffness']
    demo_trans_stiffness = data_dict['experimental_setup']['demo_trans_stiffness']
    dt = data_dict['experimental_setup']['dt']
    n_demos = len(data_dict['demo'])
    duration = data_dict['experimental_setup']['duration']
    ts = np.arange(0, duration, dt)
    N = len(ts)

    n_epochs = model_dict['n_epochs']
    #lr = model_dict['lr']
    K_min = model_dict['K_min']
    K_max = model_dict['K_max']
    hidden_layers = model_dict['hidden_layers']
    #batch_size = model_dict['batch_size']
    #l1_gain = model_dict['l1_gain']
    use_validation = model_dict['use_validation']
    model_name = str(hidden_layers[0])
    for k in range(1, len(hidden_layers)):
        model_name += '_' + str(hidden_layers[k])

    ### TODO: DELETE THIS
    print(f'Showing hyperparameters for model: {model_name}')
    print(f'EXPERIMENTAL SETUP')
    print(f"\tdt = {data_dict['experimental_setup']['dt']}")
    print(f"\tduplo_width = {data_dict['experimental_setup']['duplo_width']}")
    print(f"\tduration = {data_dict['experimental_setup']['duration']}")
    print(f"\tT_demo = {data_dict['experimental_setup']['T_demo']}")
    print(f"\trotational_stiffness = {data_dict['experimental_setup']['rotational_stiffness']}")
    print(f"\tdemo_trans_stiffness = {data_dict['experimental_setup']['demo_trans_stiffness']}")
    print(f'HYPERPARAMETERS')
    print(f"\tn_epochs = {model_dict['n_epochs']}")
    print(f"\tlr = {model_dict['lr']}")
    print(f"\tK_min = {model_dict['K_min']}")
    print(f"\tK_max = {model_dict['K_max']}")
    print(f"\thidden_layers = {model_dict['hidden_layers']}")
    print(f"\tbatch_size = {model_dict['batch_size']}")
    print(f"\tl1_gain = {model_dict['l1_gain']}")
    print(f"\tuse_validation = {model_dict['use_validation']}")
    print(f"Final train loss = {model_dict['train_losses'][-1] - model_dict['l1_losses'][-1]}")
    print(f"Final validation loss = {model_dict['val_losses'][-1] - model_dict['l1_losses'][-1]}")

    squared_error = np.zeros((n_demos - int(use_validation), N, 3))
    for j in range(n_demos - int(use_validation)):
        squared_error[j] = np.square(model_dict[j]['fc_hat'][:, :3] - data_dict['demo'][j]['hc'][:, :3])

    force_mse_train = squared_error.mean()
    force_mse_val = (np.square(model_dict[n_demos - 1]['fc_hat'][:, :3] - data_dict['demo'][n_demos - 1]['hc'][:, :3])).mean()
    print(f"Force error train = {force_mse_train}")
    print(f"Force error val = {force_mse_val}")

    squared_error_stiffness = np.zeros((n_demos - int(use_validation), N, 3))
    for j in range(n_demos - int(use_validation)):
        for i in range(3):
            squared_error_stiffness[j, :, i] = np.square(model_dict[j]['Kp_hat'][:, i] - demo_trans_stiffness[i])
    squared_error_stiffness_val = np.zeros((N, 3))
    for i in range(3):
        squared_error_stiffness_val[:, i] = np.square(model_dict[n_demos - 1]['Kp_hat'][:, i] - demo_trans_stiffness[i])

    stiffness_mse_train = squared_error_stiffness.mean()
    stiffness_mse_val = squared_error_stiffness_val.mean()
    print(f"Stiffness MSE train = {stiffness_mse_train}")
    print(f"Stiffness MSE val = {stiffness_mse_val}")

    inp = input('Continue or exit?')
    ### TODO: DELETE THIS END

    ##### PLOTS #####
    # Plot training progress
    fig, ax = plt.subplots(figsize=(8, 5))
    color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(np.arange(1, n_epochs + 1), model_dict['train_losses'],
            label=r'Training loss (including $L_1$)', color=color)
    ax.plot(np.arange(1, n_epochs + 1), model_dict['train_losses'] - model_dict['l1_losses'], '--',
            label=r'Training loss (without $L_1$)', color=color)
    color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(np.arange(1, n_epochs + 1), model_dict['val_losses'],
            label=r'Validation loss (including $L_1$)', color=color)
    ax.plot(np.arange(1, n_epochs + 1), model_dict['val_losses'] - model_dict['l1_losses'], '--',
            label=r'Validation loss (without $L_1$)', color=color)
    color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(np.arange(1, n_epochs + 1), model_dict['l1_losses'], '.-',
            label=r'Regularisation loss ($L_1$)', color=color, linewidth=0.5)
    ax.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    x1, x2, y1, y2 = 90, 100.5, 0, 1
    axins = ax.inset_axes([0.7, 0.3, 0.27, 0.27])
    color = next(axins._get_lines.prop_cycler)['color']
    axins.plot(np.arange(1, n_epochs + 1), model_dict['train_losses'], color=color, linewidth=1)
    axins.plot(np.arange(1, n_epochs + 1), model_dict['train_losses'] - model_dict['l1_losses'], '--', color=color,
               linewidth=1)
    color = next(axins._get_lines.prop_cycler)['color']
    axins.plot(np.arange(1, n_epochs + 1), model_dict['val_losses'], color=color, linewidth=1)
    axins.plot(np.arange(1, n_epochs + 1), model_dict['val_losses'] - model_dict['l1_losses'], '--', color=color,
               linewidth=1)
    color = next(axins._get_lines.prop_cycler)['color']
    axins.plot(np.arange(1, n_epochs + 1), model_dict['l1_losses'], '.-', color=color, linewidth=0.5)

    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    axins.tick_params(axis='y')
    axins.set_xticks(np.arange(x1, x2, 2))
    axins.grid()
    # axins.locator_params(tight=True, nbins=4)
    # axins.set_aspect('auto')
    ax.indicate_inset_zoom(axins, edgecolor='black')

    plt.legend()
    # plt.suptitle(f'Training progress per epoch for model: {model_name}')
    plt.tight_layout()
    if save_figures:
        plt.savefig(args.dir + f'/{n_demos - int(use_validation)}demos_training_progress_{model_name}.eps',
                    format='eps', bbox_inches='tight')
    plt.show()

    # Plot nominal trajectories
    plt.figure(figsize=(8.0, 8.0))
    plt.subplot(321)
    ax = plt.gca()
    for j in range(n_demos - int(use_validation)):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(data_dict['demo'][j]['times'], data_dict['demo'][j]['positions'][:, 0],
                 ts, data_dict['demo'][j]['target_positions'][:, 0], '--', color=color)
    plt.grid()
    plt.ylim(-0.3, 1.0)  # data sheet (-0.855, 0.855)
    plt.legend(['Actual', 'Desired'], fontsize='small')
    plt.title('Cartesian position')
    plt.ylabel(r'$x$ [m]', labelpad=-5)
    plt.subplot(323)
    ax = plt.gca()
    for j in range(n_demos - int(use_validation)):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(data_dict['demo'][j]['times'], data_dict['demo'][j]['positions'][:, 1],
                 ts, data_dict['demo'][j]['target_positions'][:, 1], '--', color=color)
    plt.grid()
    plt.ylim(-0.6, 0.7)  # data sheet: (-0.855, 0.855)
    plt.legend(['Actual', 'Desired'], fontsize='small')
    #plt.title('Trajectory y')
    plt.ylabel(r'$y$ [m]', labelpad=-5)
    plt.subplot(325)
    ax = plt.gca()
    for j in range(n_demos - int(use_validation)):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(data_dict['demo'][j]['times'], data_dict['demo'][j]['positions'][:, 2],
                 ts, data_dict['demo'][j]['target_positions'][:, 2], '--', color=color)
    plt.grid()
    plt.ylim(-0.3, 1.0)  # data sheet: (-0.36, 1.19)
    plt.legend(['Actual', 'Desired'], fontsize='small')
    #plt.title('Trajectory z')
    plt.ylabel(r'$z$ [m]', labelpad=-5)
    plt.xlabel('Time [s]')

    plt.subplot(322)
    ax = plt.gca()
    for j in range(n_demos - int(use_validation)):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(data_dict['demo'][j]['times'], data_dict['demo'][j]['angle_axis'][:, 0],
                 ts, data_dict['demo'][j]['target_angle_axis'][:, 0], '--', color=color, linewidth=1)
    plt.grid()
    plt.yticks(np.linspace(3 / 4 * np.pi, 5 / 4 * np.pi, 3), [r'$3\pi/4$', r'$\pi$', r'$5\pi/4$'])
    plt.legend(['Actual', 'Desired'], fontsize='small')
    plt.title('Orientation (rotation vectors)')
    plt.ylabel(r'$\theta \cdot v_x$', labelpad=-5)
    plt.subplot(324)
    ax = plt.gca()
    for j in range(n_demos - int(use_validation)):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(data_dict['demo'][j]['times'], data_dict['demo'][j]['angle_axis'][:, 1],
                 ts, data_dict['demo'][j]['target_angle_axis'][:, 1], '--', color=color, linewidth=1)
    plt.grid()
    plt.yticks(np.linspace(-3 / 4 * np.pi, 3 / 4 * np.pi, 7),
               [r'$-3\pi/4$', r'$-\pi/2$', r'$-\pi/4$', r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$'])
    plt.legend(['Actual', 'Desired'], fontsize='small')
    plt.ylabel(r'$\theta \cdot v_y$', labelpad=-5)
    plt.subplot(326)
    ax = plt.gca()
    for j in range(n_demos - int(use_validation)):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(data_dict['demo'][j]['times'], data_dict['demo'][j]['angle_axis'][:, 2],
                 ts, data_dict['demo'][j]['target_angle_axis'][:, 2], '--', color=color, linewidth=1)
    plt.grid()
    plt.yticks(np.linspace(-1 / 4 * np.pi, 1 / 2 * np.pi, 4), [r'$-\pi/4$', r'$0$', r'$\pi/4$', r'$\pi/2$'])
    plt.legend(['Actual', 'Desired'], fontsize='small')
    plt.ylabel(r'$\theta \cdot v_z$', labelpad=-5)
    plt.xlabel('Time [s]')
    # plt.suptitle('End effector pose (demonstration trajectories)')
    plt.tight_layout()
    if save_figures:
        plt.savefig(args.dir + f'/{n_demos-int(use_validation)}demos_ee_pose_demonstration.eps', format='eps', bbox_inches='tight')
    plt.show()

    if use_validation:
        plt.figure(figsize=(8.0, 8.0))
        plt.subplot(321)
        plt.plot(data_dict['demo'][n_demos - 1]['times'], data_dict['demo'][n_demos - 1]['positions'][:, 0],
                 ts, data_dict['demo'][n_demos - 1]['target_positions'][:, 0], '--', color='tab:red', linewidth=1)
        plt.grid()
        plt.ylim(-0.3, 1.0)  # data sheet (-0.855, 0.855)
        plt.legend(['Actual', 'Desired'], fontsize='small')
        plt.title('Cartesian position')
        plt.ylabel(r'$x$ [m]', labelpad=-5)
        plt.subplot(323)
        plt.plot(data_dict['demo'][n_demos - 1]['times'], data_dict['demo'][n_demos - 1]['positions'][:, 1],
                 ts, data_dict['demo'][n_demos - 1]['target_positions'][:, 1], '--', color='tab:red', linewidth=1)
        plt.grid()
        plt.ylim(-0.6, 0.7)  # data sheet: (-0.855, 0.855)
        plt.legend(['Actual', 'Desired'], fontsize='small')
        #plt.title('Trajectory y')
        plt.ylabel(r'$y$ [m]', labelpad=-5)
        plt.subplot(325)
        plt.plot(data_dict['demo'][n_demos - 1]['times'], data_dict['demo'][n_demos - 1]['positions'][:, 2],
                 ts, data_dict['demo'][n_demos - 1]['target_positions'][:, 2], '--', color='tab:red', linewidth=1)
        plt.grid()
        plt.ylim(-0.3, 1.0)  # data sheet: (-0.36, 1.19)
        plt.legend(['Actual', 'Desired'], fontsize='small')
        #plt.title('Trajectory z')
        plt.ylabel(r'$z$ [m]', labelpad=-5)
        plt.xlabel('Time [s]')
        plt.subplot(322)
        plt.plot(data_dict['demo'][n_demos - 1]['times'], data_dict['demo'][n_demos - 1]['angle_axis'][:, 0],
                 ts, data_dict['demo'][n_demos - 1]['target_angle_axis'][:, 0], '--', color='tab:blue', linewidth=1)
        plt.grid()
        plt.yticks(np.linspace(3 / 4 * np.pi, 5 / 4 * np.pi, 3), [r'$3\pi/4$', r'$\pi$', r'$5\pi/4$'])
        plt.legend(['Actual', 'Desired'], fontsize='small')
        plt.title('Orientation (rotation vector)')
        plt.ylabel(r'$\theta \cdot v_x$ [rad]', labelpad=-5)
        plt.subplot(324)
        plt.plot(data_dict['demo'][n_demos - 1]['times'], data_dict['demo'][n_demos - 1]['angle_axis'][:, 1],
                 ts, data_dict['demo'][n_demos - 1]['target_angle_axis'][:, 1], '--', color='tab:blue', linewidth=1)
        plt.grid()
        plt.yticks(np.linspace(-3 / 4 * np.pi, 3 / 4 * np.pi, 7),
                   [r'$-3\pi/4$', r'$-\pi/2$', r'$-\pi/4$', r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$'])
        plt.legend(['Actual', 'Desired'], fontsize='small')
        #plt.title('Trajectory y')
        plt.ylabel(r'$\theta \cdot v_y$ [rad]', labelpad=-5)
        plt.subplot(326)
        plt.plot(data_dict['demo'][n_demos - 1]['times'], data_dict['demo'][n_demos - 1]['angle_axis'][:, 2],
                 ts, data_dict['demo'][n_demos - 1]['target_angle_axis'][:, 2], '--', color='tab:blue', linewidth=1.5)
        plt.grid()
        plt.yticks(np.linspace(-1 / 4 * np.pi, 1 / 2 * np.pi, 4), [r'$-\pi/4$', r'$0$', r'$\pi/4$', r'$\pi/2$'])
        plt.legend(['Actual', 'Desired'], fontsize='small')
        #plt.title('Trajectory z')
        plt.ylabel(r'$\theta \cdot v_z$ [rad]', labelpad=-5)
        plt.xlabel('Time [s]')

        # plt.suptitle('End effector pose (validation trajectory)')
        plt.tight_layout()
        if save_figures:
            plt.savefig(args.dir + f'/ee_pose_validation.eps', format='eps', bbox_inches='tight')
        plt.show()

    # Plot stiffness trajectories
    plt.figure(figsize=(8.0, 8.0))
    plt.subplot(311)
    for j in range(n_demos - int(use_validation)):
        line, = plt.plot(ts, model_dict[j]['Kp_hat'][:, 0], linewidth=1)
        if j == 0:
            line.set_label('Predicted stiffness')
    plt.hlines(demo_trans_stiffness[0], ts[0] - duration / 20, ts[-1] + duration / 20,
               colors='r', linestyles='--', label='Target stiffness')
    plt.hlines([K_min, K_max], ts[0] - duration / 20, ts[-1] + duration / 20,
               colors='tab:green', linestyles='--', label='Stiffness limits')
    plt.grid()
    plt.xlim(ts[0] - duration/20, ts[-1] + duration/20)
    plt.ylim(0, K_max + 100)
    plt.legend(fontsize='small')
    #plt.title('Stiffness in x-direction')
    plt.ylabel(r'$k_x$')
    plt.subplot(312)
    for j in range(n_demos - int(use_validation)):
        line, = plt.plot(ts, model_dict[j]['Kp_hat'][:, 1], linewidth=1)
        if j == 0:
            line.set_label('Predicted stiffness')
    plt.hlines(demo_trans_stiffness[1], ts[0] - duration / 20, ts[-1] + duration / 20,
               colors='r', linestyles='--', label='Target stiffness')
    plt.hlines([K_min, K_max], ts[0] - duration / 20, ts[-1] + duration / 20,
               colors='tab:green', linestyles='--', label='Stiffness limits')
    plt.grid()
    plt.xlim(ts[0] - duration/20, ts[-1] + duration/20)
    plt.ylim(0, K_max + 100)
    plt.legend(fontsize='small')
    #plt.title('Stiffness in y-direction')
    plt.ylabel(r'$k_y$')
    plt.subplot(313)
    for j in range(n_demos - int(use_validation)):
        line, = plt.plot(ts, model_dict[j]['Kp_hat'][:, 2], linewidth=1)
        if j == 0:
            line.set_label('Predicted stiffness')
    plt.hlines(demo_trans_stiffness[2], ts[0] - duration / 20, ts[-1] + duration / 20,
               colors='r', linestyles='--', label='Target stiffness')
    plt.hlines([K_min, K_max], ts[0] - duration / 20, ts[-1] + duration / 20,
               colors='tab:green', linestyles='--', label='Stiffness limits')
    plt.grid()
    plt.xlim(ts[0] - duration/20, ts[-1] + duration/20)
    plt.ylim(0, K_max + 100)
    plt.legend(fontsize='small')
    #plt.title('Stiffness in z-direction')
    plt.ylabel(r'$k_z$')
    plt.xlabel('Time [s]')
    # plt.suptitle(f'Training stiffness estimates by model: {model_name}')
    #plt.tight_layout()
    if save_figures:
        plt.savefig(args.dir + f'/{n_demos-int(use_validation)}demos_stiffness_training_{model_name}.eps', format='eps', bbox_inches='tight')
    plt.show()

    if use_validation:
        plt.figure(figsize=(8.0, 8.0))
        plt.subplot(311)
        plt.plot(ts, model_dict[n_demos - 1]['Kp_hat'][:, 0], label='Predicted stiffness')
        plt.hlines(demo_trans_stiffness[0], ts[0] - duration / 20, ts[-1] + duration / 20,
                   colors='r', linestyles='--', label='Target stiffness')
        plt.hlines([K_min, K_max], ts[0] - duration / 20, ts[-1] + duration / 20,
                   colors='tab:green', linestyles='--', label='Stiffness limits')
        plt.grid()
        plt.xlim(ts[0] - duration/20, ts[-1] + duration/20)
        plt.ylim(0, K_max + 100)
        plt.legend(fontsize='small')
        #plt.title('Stiffness in x-direction')
        plt.ylabel(r'$k_x$')
        plt.subplot(312)
        plt.plot(ts, model_dict[n_demos - 1]['Kp_hat'][:, 1], label='Predicted stiffness')
        plt.hlines(demo_trans_stiffness[1], ts[0] - duration / 20, ts[-1] + duration / 20,
                   colors='r', linestyles='--', label='Target stiffness')
        plt.hlines([K_min, K_max], ts[0] - duration / 20, ts[-1] + duration / 20,
                   colors='tab:green', linestyles='--', label='Stiffness limits')
        plt.grid()
        plt.xlim(ts[0] - duration/20, ts[-1] + duration/20)
        plt.ylim(0, K_max + 100)
        plt.legend(fontsize='small')
        #plt.title('Stiffness in y-direction')
        plt.ylabel(r'$k_y$')
        plt.subplot(313)
        plt.plot(ts, model_dict[n_demos - 1]['Kp_hat'][:, 2], label='Predicted stiffness')
        plt.hlines(demo_trans_stiffness[2], ts[0] - duration / 20, ts[-1] + duration / 20,
                   colors='r', linestyles='--', label='Target stiffness')
        plt.hlines([K_min, K_max], ts[0] - duration / 20, ts[-1] + duration / 20,
                   colors='tab:green', linestyles='--', label='Stiffness limits')
        plt.grid()
        plt.xlim(ts[0] - duration/20, ts[-1] + duration/20)
        plt.ylim(0, K_max + 100)
        plt.legend(fontsize='small')
        #plt.title('Stiffness in z-direction')
        plt.ylabel(r'$k_z$')
        plt.xlabel('Time [s]')
        # plt.suptitle(f'Validation stiffness estimates by model: {model_name}')
        plt.tight_layout()
        if save_figures:
            plt.savefig(args.dir + f'/{n_demos-int(use_validation)}demos_stiffness_validation_{model_name}.eps', format='eps', bbox_inches='tight')
        plt.show()

    # Plot fc-trajectories
    plt.figure(figsize=(8.0, 8.0))
    plt.subplot(311)
    ax = plt.gca()
    for j in range(n_demos - int(use_validation)):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(ts, model_dict[j]['fc_hat'][:, 0],
                 ts, data_dict['demo'][j]['hc'][:, 0], '--', color=color, linewidth=1)
    plt.grid()
    plt.ylim(-20, 40)
    plt.legend(['Predicted force', 'Measured force'], fontsize='small')
    #plt.title('Control force in x-direction')
    plt.ylabel(r'$f_x$')
    plt.subplot(312)
    ax = plt.gca()
    for j in range(n_demos - int(use_validation)):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(ts, model_dict[j]['fc_hat'][:, 1],
                 ts, data_dict['demo'][j]['hc'][:, 1], '--', color=color, linewidth=1)
    plt.grid()
    plt.ylim(-20, 20)
    plt.legend(['Predicted force', 'Measured force'], fontsize='small')
    #plt.title('Control force in y-direction')
    plt.ylabel(r'$f_y$')
    plt.subplot(313)
    ax = plt.gca()
    for j in range(n_demos - int(use_validation)):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(ts, model_dict[j]['fc_hat'][:, 2],
                 ts, data_dict['demo'][j]['hc'][:, 2], '--', color=color, linewidth=1)
    plt.grid()
    plt.ylim(-30, 20)
    plt.legend(['Predicted force', 'Measured force'], fontsize='small')
    #plt.title('Control force in z-direction')
    plt.ylabel(r'$f_z$')
    plt.xlabel('Time [s]')
    # plt.suptitle(f'Control force estimates on demonstration trajectories by model: {model_name}')
    #plt.tight_layout()
    if save_figures:
        plt.savefig(args.dir + f'/{n_demos-int(use_validation)}demos_force_training_{model_name}.eps', format='eps', bbox_inches='tight')
    plt.show()

    if use_validation:
        plt.figure(figsize=(8.0, 8.0))
        plt.subplot(311)
        plt.plot(ts, model_dict[n_demos - 1]['fc_hat'][:, 0], linewidth=1)
        plt.plot(ts, data_dict['demo'][n_demos - 1]['hc'][:, 0], '--', linewidth=1)
        plt.grid()
        plt.ylim(-20, 40)
        plt.legend(['Predicted force', 'Measured force'], fontsize='small')
        #plt.title('Control force in x-direction')
        plt.ylabel(r'$f_x$')
        plt.subplot(312)
        plt.plot(ts, model_dict[n_demos - 1]['fc_hat'][:, 1], linewidth=1)
        plt.plot(ts, data_dict['demo'][n_demos - 1]['hc'][:, 1], '--', linewidth=1)
        plt.grid()
        plt.ylim(-20, 20)
        plt.legend(['Predicted force', 'Measured force'], fontsize='small')
        #plt.title('Control force in y-direction')
        plt.ylabel(r'$f_y$')
        plt.subplot(313)
        plt.plot(ts, model_dict[n_demos - 1]['fc_hat'][:, 2], linewidth=1)
        plt.plot(ts, data_dict['demo'][n_demos - 1]['hc'][:, 2], '--', linewidth=1)
        plt.grid()
        plt.ylim(-30, 20)
        plt.legend(['Predicted force', 'Measured force'], fontsize='small')
        #plt.title('Control force in z-direction')
        plt.ylabel(r'$f_z$')
        plt.xlabel('Time [s]')
        # plt.suptitle(f'Control force estimates on validation trajectory by model: {model_name}')
        #plt.tight_layout()
        if save_figures:
            plt.savefig(args.dir + f'/{n_demos-int(use_validation)}demos_force_validation_{model_name}.eps', format='eps', bbox_inches='tight')
        plt.show()
