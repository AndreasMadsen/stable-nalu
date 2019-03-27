
import os
import json
import torch
import stable_nalu
import itertools
import multiprocessing

use_cuda = torch.cuda.is_available()

layer_types = [
    'Tanh',
    #'Sigmoid',
    'ReLU6',
    #'Softsign',
    #'SELU',
    #'ELU',
    'ReLU',
    'linear',
    'NAC',
    'NALU'
]

operations = [
    'add',
    'sub',
    'mul',
    'div',
    'squared',
    'root'
]

seeds = range(1)

num_workers = 1  # min(8, multiprocessing.cpu_count())
max_iterations = 100000

os.makedirs("results", exist_ok=True)
fp = open('results/simple_function_static.ndjson', 'w')

for layer_type, operation, seed in itertools.product(
    layer_types, operations, seeds
):
    print(f'running layer_type: {layer_type}, operation: {operation}, seed: {seed}')

    writer = stable_nalu.writer.SummaryWriter(
        log_dir=f'runs/static/{layer_type.lower()}_{operation.lower()}_{seed}')

    # Set seed
    torch.manual_seed(seed)

    # Setup datasets
    dataset_train = iter(stable_nalu.dataset.SimpleFunctionStaticDataset.dataloader(
        operation=operation,
        batch_size=128,
        num_workers=num_workers,
        input_range=1,
        seed=seed * 3 * num_workers + 0 * num_workers,
        use_cuda=use_cuda))
    dataset_valid_interpolation = iter(stable_nalu.dataset.SimpleFunctionStaticDataset.dataloader(
        operation=operation,
        batch_size=2048,
        num_workers=num_workers,
        input_range=1,
        seed=seed * 3 * num_workers + 1 * num_workers,
        use_cuda=use_cuda))
    dataset_valid_extrapolation = iter(stable_nalu.dataset.SimpleFunctionStaticDataset.dataloader(
        operation=operation,
        batch_size=2048,
        num_workers=num_workers,
        input_range=5,
        seed=seed * 3 * num_workers + 2 * num_workers,
        use_cuda=use_cuda))

    # setup model
    model = stable_nalu.network.SimpleFunctionStaticNetwork(layer_type)
    if use_cuda:
        model.cuda()
    model.reset_parameters()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Train model
    for epoch_i, (x_train, t_train) in zip(range(max_iterations + 1), dataset_train):
        writer.set_iteration(epoch_i)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        y_train = model(x_train)
        loss_train = criterion(y_train, t_train)

        # Log loss
        writer.add_scalar('loss/train', loss_train)
        if epoch_i % 100 == 0:
            x_valid_inter, t_valid_inter = next(dataset_valid_interpolation)
            loss_valid_inter = criterion(model(x_valid_inter), t_valid_inter)
            writer.add_scalar('loss/valid/interpolation', loss_valid_inter)

            x_valid_extra, t_valid_extra = next(dataset_valid_extrapolation)
            loss_valid_extra = criterion(model(x_valid_extra), t_valid_extra)
            writer.add_scalar('loss/valid/extrapolation', loss_valid_extra)

        if epoch_i % 1000 == 0:
            print(f'  {epoch_i}: {loss_train}')

        # Backward + optimize model
        loss_train.backward()
        optimizer.step()

    # Write results for this training
    print(f'  finished:')
    print(f'  - epochs: {epoch_i}')
    print(f'  - loss_train: {loss_train}')
    print(f'  - loss_valid_inter: {loss_valid_inter}')
    print(f'  - loss_valid_extra: {loss_valid_extra}')
    fp.write(json.dumps({
        'layer_type': layer_type,
        'operation': operation,
        'seed': seed,
        'epochs': epoch_i,
        'loss_train': loss_train.detach().cpu().numpy().item(0),
        'loss_valid_inter': loss_valid_inter.detach().cpu().numpy().item(0),
        'loss_valid_extra': loss_valid_extra.detach().cpu().numpy().item(0)
    }) + '\n')
    fp.flush()

    writer.close()

fp.close()
