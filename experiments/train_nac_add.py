
import numpy as np
import torch
import grumbel_nalu

dataset_train = grumbel_nalu.dataset.SimpleFunctionStaticDataset(operation='add', input_range=5, seed=0)
batch_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=64,
    shuffle=False,
    sampler=torch.utils.data.SequentialSampler(dataset_train))

model = grumbel_nalu.network.SimpleFunctionStaticNetwork('NAC')
model.reset_parameters()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch_i, (x_train, t_train) in zip(range(10000), batch_train):
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    y_train = model(x_train)
    loss = criterion(y_train, t_train)
    loss.backward()
    optimizer.step()

    if epoch_i % 10 == 0:
        print(f'{epoch_i}: {loss.item()}')

