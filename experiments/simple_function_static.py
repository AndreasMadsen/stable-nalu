
import math
import torch
import stable_nalu
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Runs the simple function static task')
parser.add_argument('--layer-type',
                    action='store',
                    default='NALU',
                    choices=list(stable_nalu.network.SimpleFunctionStaticNetwork.UNIT_NAMES),
                    type=str,
                    help='Specify the layer type, e.g. Tanh, ReLU, NAC, NALU')
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
parser.add_argument('--min-input',
                    action='store',
                    default=1,
                    type=int,
                    help='Specify the smallest possible input value')
parser.add_argument('--max-iterations',
                    action='store',
                    default=100000,
                    type=int,
                    help='Specify the max number of iterations to use')
parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help=f'Force no CUDA (cuda usage is detected automatically as {torch.cuda.is_available()})')
parser.add_argument('--nalu-bias',
                    action='store_true',
                    default=False,
                    help='Enables bias in the NALU gate')
parser.add_argument('--nalu-two-nac',
                    action='store_true',
                    default=False,
                    help='Uses two independent NACs in the NALU Layer')
parser.add_argument('--nalu-mul',
                    action='store',
                    default='normal',
                    choices=['normal', 'safe', 'trig'],
                    help='Multplication unit, can be normal, safe, trig')
parser.add_argument('--nalu-gate',
                    action='store',
                    default='normal',
                    choices=['normal', 'regualized', 'obs-gumbel', 'gumbel'],
                    type=str,
                    help='Can be normal, regualized, obs-gumbel, or gumbel')
parser.add_argument('--simple',
                    action='store_true',
                    default=False,
                    help='Use a very simple dataset with t = sum(v[0:2]) + sum(v[4:6])')
parser.add_argument('--name-prefix',
                    action='store',
                    default='simple_function_static',
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
print(f'  - seed: {args.seed}')
print(f'  - min_input: {args.min_input}')
print(f'  - operation: {args.operation}')
print(f'  - layer_type: {args.layer_type}')
print(f'  - nalu_bias: {args.nalu_bias}')
print(f'  - nalu_two_nac: {args.nalu_two_nac}')
print(f'  - nalu_mul: {args.nalu_mul}')
print(f'  - nalu_gate: {args.nalu_gate}')
print(f'  - simple: {args.simple}')
print(f'  - cuda: {args.cuda}')
print(f'  - verbose: {args.verbose}')
print(f'  - name_prefix: {args.name_prefix}')
print(f'  - max_iterations: {args.max_iterations}')

# Prepear logging
results_writer = stable_nalu.writer.ResultsWriter(args.name_prefix)
summary_writer = stable_nalu.writer.SummaryWriter(
    f'{args.name_prefix}/{args.layer_type.lower()}'
    f'{"-" if (args.nalu_bias or args.nalu_two_nac or args.nalu_mul != "normal" or args.nalu_gate != "normal") else ""}'
    f'{"b" if args.nalu_bias else ""}'
    f'{"2" if args.nalu_two_nac else ""}'
    f'{"s" if args.nalu_mul == "safe" else ""}'
    f'{"t" if args.nalu_mul == "trig" else ""}'
    f'{"r" if args.nalu_gate == "regualized" else ""}'
    f'{"g" if args.nalu_gate == "gumbel" else ""}'
    f'{"gg" if args.nalu_gate == "obs-gumbel" else ""}'
    f'_{args.operation.lower()}'
    f'{f"_i{args.min_input}" if args.min_input != 1 else ""}'
    f'_{args.seed}',
    remove_existing_data=args.remove_existing_data
)

# Set seed
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

# Setup datasets
dataset = stable_nalu.dataset.SimpleFunctionStaticDataset(
    operation=args.operation,
    min_input=args.min_input,
    simple=args.simple,
    use_cuda=args.cuda,
    seed=args.seed,
)
print(f'  - dataset: {dataset.print_operation()}')
dataset_train = iter(dataset.fork(input_range=1).dataloader(batch_size=128))
dataset_valid_interpolation = iter(dataset.fork(input_range=1).dataloader(batch_size=2048))
dataset_valid_extrapolation = iter(dataset.fork(input_range=5).dataloader(batch_size=2048))

# setup model
model = stable_nalu.network.SimpleFunctionStaticNetwork(
    args.layer_type,
    input_size=dataset.get_input_size(),
    writer=summary_writer.every(1000) if args.verbose else None,
    nalu_bias=args.nalu_bias,
    nalu_two_nac=args.nalu_two_nac,
    nalu_mul=args.nalu_mul,
    nalu_gate=args.nalu_gate,
)
model.reset_parameters()
if args.cuda:
    model.cuda()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

def test_model(dataloader):
    with torch.no_grad(), model.no_internal_logging(), model.no_random():
        x, t = next(dataloader)
        return torch.sqrt(criterion(model(x), t))

# Train model
for epoch_i, (x_train, t_train) in zip(range(args.max_iterations + 1), dataset_train):
    summary_writer.set_iteration(epoch_i)

    # Prepear model
    model.set_parameter('tau', max(0.5, math.exp(-1e-5 * epoch_i)))
    optimizer.zero_grad()

    # Log validation
    if epoch_i % 1000 == 0:
        interpolation_error = test_model(dataset_valid_interpolation)
        extrapolation_error = test_model(dataset_valid_extrapolation)

        summary_writer.add_scalar('loss/valid/interpolation', interpolation_error)
        summary_writer.add_scalar('loss/valid/extrapolation', extrapolation_error)

    # forward
    y_train = model(x_train)
    loss_train_criterion = torch.sqrt(criterion(y_train, t_train))
    loss_train_regualizer = 0.1 * (1 - math.exp(-1e-5 * epoch_i)) * model.regualizer()
    loss_train = loss_train_criterion + loss_train_regualizer

    # Log loss
    summary_writer.add_scalar('loss/train/critation', loss_train_criterion)
    summary_writer.add_scalar('loss/train/regualizer', loss_train_regualizer)
    summary_writer.add_scalar('loss/train/total', loss_train)
    if epoch_i % 1000 == 0:
        print('train %d: %.5f, inter: %.5f, extra: %.5f' % (epoch_i, loss_train_criterion, interpolation_error, extrapolation_error))

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
    'min_input': args.min_input,
    'operation': args.operation,
    'layer_type': args.layer_type,
    'nalu_bias': args.nalu_bias,
    'nalu_two_nac': args.nalu_two_nac,
    'nalu_mul': args.nalu_mul,
    'nalu_gate': args.nalu_gate,
    'simple': args.simple,
    'cuda': args.cuda,
    'verbose': args.verbose,
    'name_prefix': args.name_prefix,
    'max_iterations': args.max_iterations,
    'loss_train': loss_train,
    'loss_valid_inter': loss_valid_inter,
    'loss_valid_extra': loss_valid_extra,
})
