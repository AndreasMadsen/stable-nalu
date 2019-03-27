
import numpy as np
import torch
import stable_nalu

batch_train = stable_nalu.dataset.SimpleFunctionStaticDataset.dataloader(
    operation='add',
    batch_size=128,
    num_workers=0,
    input_range=1
)

model = stable_nalu.network.SimpleFunctionStaticNetwork('NAC')
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

