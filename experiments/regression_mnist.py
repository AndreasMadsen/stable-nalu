
import os
import ast
import math
import torch
import stable_nalu
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Run either the MNIST counting or MNIST Arithmetic task')
parser.add_argument('--max-epochs',
                    action='store',
                    default=1000,
                    type=int,
                    help='Specify the max number of epochs to use')
parser.add_argument('--batch-size',
                    action='store',
                    default=64,
                    type=int,
                    help='Specify the batch-size to be used for training')
parser.add_argument('--seed',
                    action='store',
                    default=0,
                    type=int,
                    help='Specify the seed to use')

parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help=f'Force no CUDA (cuda usage is detected automatically as {torch.cuda.is_available()})')
parser.add_argument('--name-prefix',
                    action='store',
                    default='regression_mnist',
                    type=str,
                    help='Where the data should be stored')
parser.add_argument('--remove-existing-data',
                    action='store_true',
                    default=False,
                    help='Should old results be removed')
parser.add_argument('--verbose',
                    action='store_true',
                    default=False,
                    help='Should network measures (e.g. gates) and gradients be shown')
args = parser.parse_args()

setattr(args, 'cuda', torch.cuda.is_available() and not args.no_cuda)

# Print configuration
print(f'running')
print(f'  - max_epochs: {args.max_epochs}')
print(f'  - batch_size: {args.batch_size}')
print(f'  - seed: {args.seed}')
print(f'  -')
print(f'  - cuda: {args.cuda}')
print(f'  - name_prefix: {args.name_prefix}')
print(f'  - remove_existing_data: {args.remove_existing_data}')
print(f'  - verbose: {args.verbose}')

# Prepear logging
summary_writer = stable_nalu.writer.SummaryWriter(
    f'{args.name_prefix}/regresssion'
    f'_b{args.batch_size}'
    f'_s{args.seed}',
    remove_existing_data=args.remove_existing_data
)

# Set threads
if 'LSB_DJOB_NUMPROC' in os.environ:
    torch.set_num_threads(int(os.environ['LSB_DJOB_NUMPROC']))

# Set seed
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

# Setup datasets
dataset = stable_nalu.dataset.SequentialMnistDataset(
    operation='sum',
    use_cuda=args.cuda,
    seed=args.seed
)
dataset_train = dataset.fork(seq_length=1, subset='train').dataloader(shuffle=True)
# Seeds are from random.org
dataset_valid = dataset.fork(seq_length=1, subset='train',
                             seed=62379872).dataloader(shuffle=False)
dataset_test = dataset.fork(seq_length=1, subset='test',
                            seed=47430696).dataloader(shuffle=False)

# setup model
model = stable_nalu.network.RegressionMnisNetwork()
model.reset_parameters()
if args.cuda:
    model.cuda()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

def test_model(dataloader):
    with torch.no_grad(), model.no_internal_logging(), model.no_random():
        acc_loss = 0
        for x, t in dataloader:
            # forward
            y = model(x[:, 0, ...])
            acc_loss += criterion(y, t).item() * len(t)

        return acc_loss / len(dataloader.dataset)

def summarize_model(dataloader):
    with torch.no_grad(), model.no_internal_logging(), model.no_random():
        acc_loss = 0
        y = torch.cat([model(x[:, 0, ...]) for x, t in dataloader])

        return(torch.mean(y, 0), torch.var(y, 0))

# Train model
global_step = 0
for epoch_i in range(0, args.max_epochs + 1):
    for i_train, (x_train, t_train) in enumerate(dataset_train):
        global_step += 1
        summary_writer.set_iteration(global_step)
        summary_writer.add_scalar('epoch', epoch_i)

        # Prepear model
        optimizer.zero_grad()

        # Log validation
        if epoch_i % 100 == 0 and i_train == 0:
            valid_error = test_model(dataset_valid)
            test_error = test_model(dataset_test)
            summary_writer.add_scalar('loss/valid', valid_error)
            summary_writer.add_scalar('loss/test', test_error)

        # forward
        y_train = model(x_train[:, 0, ...])

        loss_train = criterion(y_train, t_train)

        # Log loss
        summary_writer.add_scalar('loss/train', loss_train)
        if epoch_i % 100 == 0 and i_train == 0:
            print('train %d: %.5f, valid: %.5f, test: %.5f' % (epoch_i, loss_train, valid_error, test_error))

        # Optimize model
        if loss_train.requires_grad:
            loss_train.backward()
            optimizer.step()

        # Log gradients if in verbose mode
        if args.verbose:
            model.log_gradients()

# Write results for this training
print(f'finished:')
print(f'  - loss_train: {loss_train}')
print(f'  - valid: {valid_error}')
print(f'  - test: {test_error}')

valid_mean, valid_var = summarize_model(dataset_valid)
test_mean, test_var = summarize_model(dataset_test)
print(f'  - valid-mean: {valid_mean}, valid-var: {valid_var}')
print(f'  - test-mean: {test_mean}, test-var: {test_var}')

# Use saved weights to visualize the intermediate values.
stable_nalu.writer.save_model(summary_writer.name, model)
