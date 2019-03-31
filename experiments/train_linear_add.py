
import numpy as np
import torch
import stable_nalu

dataset = stable_nalu.dataset.SimpleFunctionStaticDataset(
    operation='add',
    use_cuda=False,
    seed=0
)
dataset_train = iter(dataset.fork(input_range=1).dataloader(batch_size=128))

model = stable_nalu.network.SimpleFunctionStaticNetwork('linear')
model.reset_parameters()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch_i, (x_train, t_train) in zip(range(1000), dataset_train):
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    y_train = model(x_train)
    loss = criterion(y_train, t_train)
    loss.backward()
    optimizer.step()

    if epoch_i % 10 == 0:
        print(f'{epoch_i}: {loss.item()}')

