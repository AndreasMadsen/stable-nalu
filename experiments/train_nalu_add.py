
import sys
import numpy as np
import torch
import stable_nalu

seed = int(sys.argv[1])
use_cuda = torch.cuda.is_available()
torch.manual_seed(seed)

print(f'running with seed: {seed}')

writer = stable_nalu.writer.SummaryWriter(log_dir=f'runs/repeat/static/nalu/add/seed_{seed}')
dataset = stable_nalu.dataset.SimpleFunctionStaticDataset(
    operation='add',
    use_cuda=use_cuda,
    num_workers=1,
    seed=seed
)
dataset_train = iter(dataset.fork(input_range=1).dataloader(batch_size=128))
dataset_valid_interpolation = iter(dataset.fork(input_range=1).dataloader(batch_size=2048))
dataset_valid_extrapolation = iter(dataset.fork(input_range=5).dataloader(batch_size=2048))

model = stable_nalu.network.SimpleFunctionStaticNetwork('NALU', writer=writer.namespace('network'))
model.reset_parameters()
if use_cuda:
    model.cuda()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch_i, (x_train, t_train) in zip(range(100000), dataset_train):
    writer.set_iteration(epoch_i)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    y_train = model(x_train)
    loss_train = criterion(y_train, t_train)
    loss_train.backward()

    if np.isnan(loss_train.detach().cpu().numpy().item(0)):
        break

    writer.add_summary('train/y', y_train)
    writer.add_scalar('loss/train', loss_train)
    if epoch_i % 100 == 0:
        print(f'{epoch_i}: {loss_train.item()}')

        # Log loss
        x_valid_inter, t_valid_inter = next(dataset_valid_interpolation)
        loss_valid_inter = criterion(model(x_valid_inter), t_valid_inter)
        writer.add_scalar('loss/valid/interpolation', loss_valid_inter)

        x_valid_extra, t_valid_extra = next(dataset_valid_extrapolation)
        loss_valid_extra = criterion(model(x_valid_extra), t_valid_extra)
        writer.add_scalar('loss/valid/extrapolation', loss_valid_extra)

        # Log weights
        for index, weight in enumerate(model.parameters(), start=1):
            gradient, *_ = weight.grad.data
            writer.add_summary(f'grad/{index}', gradient)

    optimizer.step()
