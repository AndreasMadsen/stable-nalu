
import numpy as np
import torch
import stable_nalu

def test_linear_add_can_have_zero_loss():
    # Prepear data
    dataset = stable_nalu.dataset.SimpleFunctionStaticDataset(
        operation='add',
        seed=0
    )
    dataset_eval = iter(dataset.fork(input_range=1).dataloader(batch_size=128))

    # Setup pre-solved model
    model = stable_nalu.network.SimpleFunctionStaticNetwork('linear')

    w_1 = np.zeros((100, 2), dtype=np.float32)
    w_1[dataset.a_start:dataset.a_end, 0] = 1
    w_1[dataset.b_start:dataset.b_end, 1] = 1
    w_2 = np.ones((2, 1), dtype=np.float32)

    model.layer_1.layer.weight.data = torch.tensor(np.transpose(w_1))
    model.layer_2.layer.weight.data = torch.tensor(np.transpose(w_2))

    # Compute loss
    criterion = torch.nn.MSELoss()
    for i, (x_train, t_train) in zip(range(5), dataset_eval):
        y_train = model(x_train)
        loss = criterion(y_train, t_train)
        np.testing.assert_almost_equal(
            loss.detach().numpy(),
            0
        )

def test_linear_add_is_trainable():
    # Prepear data
    dataset = stable_nalu.dataset.SimpleFunctionStaticDataset(
        operation='add',
        seed=0
    )
    dataset_train = iter(dataset.fork(input_range=1).dataloader(batch_size=128))

    torch.manual_seed(0)
    model = stable_nalu.network.SimpleFunctionStaticNetwork('linear')
    model.reset_parameters()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for epoch_i, (x_train, t_train) in zip(range(200), dataset_train):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        y_train = model(x_train)
        loss = criterion(y_train, t_train)
        loss.backward()
        optimizer.step()

    # Check that last loss is 0
    np.testing.assert_almost_equal(
        loss.detach().numpy(),
        0,
        decimal=5
    )
