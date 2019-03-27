
import numpy as np
import torch
import stable_nalu

writer = stable_nalu.writer.SummaryWriter(log_dir='runs/debug/nalu')
batch_train = stable_nalu.dataset.SimpleFunctionStaticDataset.dataloader(
    operation='add',
    batch_size=128,
    num_workers=0,
    input_range=1,
    seed=0
)
model = stable_nalu.network.SimpleFunctionStaticNetwork('NALU', writer=writer.namespace('network'))
model.reset_parameters()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch_i, (x_train, t_train) in zip(range(100000), batch_train):
    writer.set_iteration(epoch_i)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    y_train = model(x_train)
    writer.add_summary('train/y', y_train)
    loss = criterion(y_train, t_train)
    writer.add_scalar('train/loss', loss)
    loss.backward()

    if np.isnan(loss.detach().cpu().numpy().item(0)):
        break

    if epoch_i % 100 == 0:
        print(f'{epoch_i}: {loss.item()}')

        for index, weight in enumerate(model.parameters(), start=1):
            gradient, *_ = weight.grad.data
            writer.add_summary(f'grad/{index}', gradient)

        torch.save({
            'epoch': epoch_i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, 'weights/debug/nalu')

    optimizer.step()
