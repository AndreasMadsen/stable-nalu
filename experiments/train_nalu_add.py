
import numpy as np
import torch
import stable_nalu

writer = stable_nalu.writer.SummaryWriter(log_dir='runs/nalu')
dataset_train = stable_nalu.dataset.SimpleFunctionStaticDataset(operation='add', input_range=5, seed=0)
batch_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=128,
    shuffle=False,
    sampler=torch.utils.data.SequentialSampler(dataset),
    num_workers=num_workers,
    worker_init_fn=dataset.worker_init_fn)

model = stable_nalu.network.SimpleFunctionStaticNetwork('NALU', writer=writer.namespace('network'))
model.reset_parameters()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

for epoch_i, (x_train, t_train) in zip(range(100000), batch_train):
    writer.set_iteration(epoch_i)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    y_train = model(x_train)
    loss = criterion(y_train, t_train)
    writer.add_scalar('loss', loss)
    loss.backward()
    optimizer.step()

    for index, weight in enumerate(model.parameters(), start=1):
        gradient, *_ = weight.grad.data
        writer.add_summary(f'grad/{index}', gradient)

    if epoch_i % 100 == 0:
        print(f'{epoch_i}: {loss.item()}')

