import numpy as np
import torch
from frankapy.nn import NeuralNet


if __name__ == "__main__":
    # Experiment parameters:
    USE_GPU = True
    dt = 0.01
    T_demo = 8.
    n_demos = 1
    T_insert = 1.  # The time allocated for final insertion phase
    T_converge = 1.  # The time allocated for convergence after insertion
    duration = T_demo + T_insert + T_converge
    rotational_stiffness = [30.] * 3
    demo_trans_stiffness = [800.] * 3  # high stiffness for demonstrations
    print_every = 25

    # Tunable parameters:
    init_trans_stiffness_vic = [600.] * 3
    n_epochs = 100
    lr = 0.001  # Old 0.002
    K_min = 0.1
    K_max = 1500
    hidden_layers = [64, 32]  # Also try [256, 128], [256, 128, 64], [128, 64, 32] and [64, 32, 16]
    batch_size = 64
    l1_gain = 1e-3
    dtype = torch.float32
    device = torch.device('cuda') if USE_GPU and torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)

    # Initialise and train neural network for n_epochs
    model = NeuralNet(rotational_stiffness=rotational_stiffness, K_min=K_min, K_max=K_max, hidden_layers=hidden_layers,
                      dtype=dtype, device=device).to(device)

    print(f'Model.dtype = {model.dtype}')
    # Test inference
    model.eval()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    dummy_input = torch.randn((1, 30), dtype=dtype, device=device)
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_inf = np.sum(timings) / repetitions
    std_inf = np.std(timings)
    print(f'Mean infererence time across {repetitions} repetitions: {mean_inf}s, standard deviation: {std_inf}s')

    for _ in range(10):
        _ = model(dummy_input)
    with torch.no_grad():
        starter.record()
        for rep in range(repetitions):
            _ = model(dummy_input)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        total_time = starter.elapsed_time(ender)
    mean_inf = total_time/repetitions
    print(f'Mean inference time without synchronization every time across {repetitions} repetitions: {mean_inf}')