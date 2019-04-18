
import math
import torch
import stable_nalu
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Runs the simple function static task')
parser.add_argument('--layer-type',
                    action='store',
                    default='NALU',
                    choices=list(stable_nalu.network.SimpleFunctionRecurrentNetwork.UNIT_NAMES),
                    type=str,
                    help='Specify the layer type, e.g. RNN-tanh, LSTM, NAC, NALU')
parser.add_argument('--operation',
                    action='store',
                    default='add',
                    choices=[
                        'add', 'sub', 'mul', 'div', 'squared', 'root'
                    ],
                    type=str,
                    help='Specify the operation to use, e.g. add, mul, squared')
parser.add_argument('--seed',
                    action='store',
                    default=0,
                    type=int,
                    help='Specify the seed to use')
parser.add_argument('--max-iterations',
                    action='store',
                    default=100000,
                    type=int,
                    help='Specify the max number of iterations to use')
parser.add_argument('--cuda',
                    action='store',
                    default=torch.cuda.is_available(),
                    type=bool,
                    help=f'Should CUDA be used (detected automatically as {torch.cuda.is_available()})')
parser.add_argument('--simple',
                    action='store_true',
                    default=False,
                    help='Use a very simple dataset with t = sum(v[0:2]) + sum(v[4:6])')
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
print(f'  - simple: {args.simple}')
print(f'  - cuda: {args.cuda}')
print(f'  - verbose: {args.verbose}')
print(f'  - max_iterations: {args.max_iterations}')

# Prepear logging
results_writer = stable_nalu.writer.ResultsWriter('simple_function_recurrent')
summary_writer = stable_nalu.writer.SummaryWriter(
    f'simple_function_recurrent/{args.layer_type.lower()}_{args.operation.lower()}_{args.seed}'
)

# Set seed
torch.manual_seed(args.seed)

# Setup datasets
dataset = stable_nalu.dataset.SimpleFunctionRecurrentDataset(
    operation=args.operation,
    simple=args.simple,
    use_cuda=args.cuda,
    seed=args.seed
)
dataset_train = iter(dataset.fork(seq_length=10).dataloader(batch_size=128))
dataset_valid_interpolation = iter(dataset.fork(seq_length=10).dataloader(batch_size=2048))
dataset_valid_extrapolation = iter(dataset.fork(seq_length=1000).dataloader(batch_size=2048))

# setup model
model = stable_nalu.network.SimpleFunctionRecurrentNetwork(
    args.layer_type,
    writer=summary_writer.every(1000) if args.verbose else None
)
if args.cuda:
    model.cuda()
model.reset_parameters()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

def test_model(dataloader):
    with torch.no_grad(), model.no_internal_logging():
        x, t = next(dataloader)
        return criterion(model(x), t)

# Train model
for epoch_i, (x_train, t_train) in zip(range(args.max_iterations + 1), dataset_train):
    summary_writer.set_iteration(epoch_i)

    # Prepear model
    model.set_parameter('tau', max(0.5, math.exp(-1e-5 * epoch_i)))
    optimizer.zero_grad()

    # Log validation
    if epoch_i % 1000 == 0:
        summary_writer.add_scalar('loss/valid/interpolation', test_model(dataset_valid_interpolation))
        summary_writer.add_scalar('loss/valid/extrapolation', test_model(dataset_valid_extrapolation))

    # forward
    y_train = model(x_train)
    loss_train_criterion = criterion(y_train, t_train)
    loss_train_regualizer = 0.1 * (1 - math.exp(-1e-5 * epoch_i)) * model.regualizer()
    loss_train = loss_train_criterion + loss_train_regualizer

    # Log loss
    summary_writer.add_scalar('loss/train/critation', loss_train_criterion)
    summary_writer.add_scalar('loss/train/regualizer', loss_train_regualizer)
    summary_writer.add_scalar('loss/train/total', loss_train)
    if epoch_i % 1000 == 0:
        print(f'train {epoch_i}: {loss_train_criterion}')

    # Optimize model
    if loss_train.requires_grad:
        loss_train.backward()
        optimizer.step()
    model.optimize(loss_train_criterion)

    # Log gradients if in verbose mode
    if args.verbose and epoch_i % 1000 == 0:
        model.log_gradients()

# Compute validation loss
loss_valid_inter = test_model(dataset_valid_interpolation)
loss_valid_extra = test_model(dataset_valid_extrapolation)

# Write results for this training
print(f'finished:')
print(f'  - loss_train: {loss_train}')
print(f'  - loss_valid_inter: {loss_valid_inter}')
print(f'  - loss_valid_extra: {loss_valid_extra}')

# save results
results_writer.add({
    'seed': args.seed,
    'operation': args.operation,
    'layer_type': args.layer_type,
    'simple': args.simple,
    'cuda': args.cuda,
    'verbose': args.verbose,
    'max_iterations': args.max_iterations,
    'loss_train': loss_train,
    'loss_valid_inter': loss_valid_inter,
    'loss_valid_extra': loss_valid_extra
})
