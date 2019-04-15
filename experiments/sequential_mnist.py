
import torch
import stable_nalu
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Run either the MNIST counting or MNIST Arithmetic task')
parser.add_argument('--layer-type',
                    action='store',
                    default='NALU',
                    choices=[
                        'RNN-tanh', 'RNN-ReLU', 'GRU', 'LSTM', 'NAC', 'NALU'
                    ],
                    type=str,
                    help='Specify the layer type, e.g. RNN-tanh, LSTM, NAC, NALU')
parser.add_argument('--operation',
                    action='store',
                    default='sum',
                    choices=[
                        'sum', 'count'
                    ],
                    type=str,
                    help='Specify the operation to use, sum or count')
parser.add_argument('--seed',
                    action='store',
                    default=0,
                    type=int,
                    help='Specify the seed to use')
parser.add_argument('--max-epochs',
                    action='store',
                    default=1000,
                    type=int,
                    help='Specify the max number of epochs to use')
parser.add_argument('--cuda',
                    action='store',
                    default=torch.cuda.is_available(),
                    type=bool,
                    help='Should CUDA be used')
parser.add_argument('--verbose',
                    action='store_true',
                    default=False,
                    help='Should network measures (e.g. gates) and gradients be shown')
args = parser.parse_args()

# Print configuration
print(f'running')
print(f'  - seed: {args.seed}')
print(f'  - operation: {args.operation}')
print(f'  - layer_type: {args.layer_type}')
print(f'  - cuda: {args.cuda}')
print(f'  - verbose: {args.verbose}')
print(f'  - max_epochs: {args.max_epochs}')

# Prepear logging
results_writer = stable_nalu.writer.ResultsWriter('sequential_mnist')
summary_writer = stable_nalu.writer.SummaryWriter(
    f'sequential_mnist/{args.layer_type.lower()}_{args.operation.lower()}_{args.seed}'
)

# Set seed
torch.manual_seed(args.seed)

# Setup datasets
dataset = stable_nalu.dataset.SequentialMnistDataset(
    operation=args.operation,
    use_cuda=args.cuda,
    seed=args.seed
)
dataset_train = dataset.fork(seq_length=10).dataloader()
dataset_valid_interpolation = dataset.fork(seq_length=10).dataloader()
dataset_valid_extrapolation_100 = dataset.fork(seq_length=100).dataloader()
dataset_valid_extrapolation_1000 = dataset.fork(seq_length=1000).dataloader()

# setup model
model = stable_nalu.network.SequentialMnistNetwork(
    args.layer_type,
    dataset.get_item_shape().target[-1],
    writer=summary_writer if args.verbose else None
)
if args.cuda:
    model.cuda()
model.reset_parameters()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

def test_model(dataloader):
    acc_loss = 0
    for x, t in dataloader:
        # forward
        y = model(x)
        acc_loss += criterion(y, t).item() * len(t)

    return acc_loss / len(dataloader.dataset)

# Train model
global_step = 0
for epoch_i in range(0, args.max_epochs + 1):
    for i_train, (x_train, t_train) in enumerate(dataset_train):
        global_step += 1
        summary_writer.set_iteration(global_step)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        y_train = model(x_train)
        loss_train = criterion(y_train, t_train)

        # Log loss
        if i_train % 10 == 0:
            print(f'train {epoch_i} [{round(i_train / len(dataset_train) * 100)}%]: {loss_train}')
        summary_writer.add_scalar('loss/train', loss_train)

        if epoch_i % 10 == 0 and i_train == 0:
            loss_valid_interpolation = test_model(dataset_valid_interpolation)
            loss_valid_extrapolation_100 = test_model(dataset_valid_extrapolation_100)
            loss_valid_extrapolation_1000 = test_model(dataset_valid_extrapolation_1000)

            summary_writer.add_scalar('loss/valid/interpolation', loss_valid_interpolation)
            summary_writer.add_scalar('loss/valid/extrapolation/100', loss_valid_extrapolation_100)
            summary_writer.add_scalar('loss/valid/extrapolation/1000', loss_valid_extrapolation_1000)

        # Backward + optimize model
        loss_train.backward()
        optimizer.step()

# Write results for this training
print(f'finished:')
print(f'  - loss_train: {loss_train}')
print(f'  - loss_valid_inter: {loss_valid_interpolation}')
print(f'  - loss_valid_extra_100: {loss_valid_extrapolation_100}')
print(f'  - loss_valid_extra_1000: {loss_valid_extrapolation_1000}')

# save results
results_writer.add({
    'seed': args.seed,
    'operation': args.operation,
    'layer_type': args.layer_type,
    'cuda': args.cuda,
    'verbose': args.verbose,
    'max_epochs': args.max_epochs,
    'loss_train': loss_train,
    'loss_valid_interpolation': loss_valid_interpolation,
    'loss_valid_extrapolation_100': loss_valid_extrapolation_100,
    'loss_valid_extrapolation_1000': loss_valid_extrapolation_1000
})
