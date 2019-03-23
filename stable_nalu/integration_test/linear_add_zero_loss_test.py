
import numpy as np
import torch
import stable_nalu

def test_linear_add_can_have_zero_loss():
    # Prepear data
    dataset_train = stable_nalu.dataset.SimpleFunctionStaticDataset(operation='add', input_range=5)
    batch_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=64,
        shuffle=False,
        sampler=torch.utils.data.SequentialSampler(dataset_train))

    # Setup pre-solved model
    model = stable_nalu.network.SimpleFunctionStaticNetwork('linear')

    w_1 = np.zeros((100, 2), dtype=np.float32)
    w_1[dataset_train.a_start:dataset_train.a_end, 0] = 1
    w_1[dataset_train.b_start:dataset_train.b_end, 1] = 1
    w_2 = np.ones((2, 1), dtype=np.float32)

    model.layer_1.layer.weight.data = torch.tensor(np.transpose(w_1))
    model.layer_2.layer.weight.data = torch.tensor(np.transpose(w_2))

    # Compute loss
    criterion = torch.nn.MSELoss()
    for i, (x_train, t_train) in zip(range(5), batch_train):
        y_train = model(x_train)
        loss = criterion(y_train, t_train)
        np.testing.assert_almost_equal(
            loss.detach().numpy(),
            0
        )

def test_linear_add_is_trainable():
    dataset_train = stable_nalu.dataset.SimpleFunctionStaticDataset(
        operation='add',
        input_range=5,
        seed=0)
    batch_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=64,
        shuffle=False,
        sampler=torch.utils.data.SequentialSampler(dataset_train))

    torch.manual_seed(0)
    model = stable_nalu.network.SimpleFunctionStaticNetwork('linear')
    model.reset_parameters()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for epoch_i, (x_train, t_train) in zip(range(100), batch_train):
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
        0
    )
