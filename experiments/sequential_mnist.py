
import os
import ast
import math
import torch
import stable_nalu
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Run either the MNIST counting or MNIST Arithmetic task')
parser.add_argument('--layer-type',
                    action='store',
                    default='NALU',
                    choices=list(stable_nalu.network.SequentialMnistNetwork.UNIT_NAMES),
                    type=str,
                    help='Specify the layer type, e.g. RNN-tanh, LSTM, NAC, NALU')
parser.add_argument('--operation',
                    action='store',
                    default='cumsum',
                    choices=[
                        'cumsum', 'sum', 'cumprod', 'prod', 'count'
                    ],
                    type=str,
                    help='Specify the operation to use, sum or count')
parser.add_argument('--regualizer',
                    action='store',
                    default=0.1,
                    type=float,
                    help='Specify the regualization lambda to be used')
parser.add_argument('--remove-zero',
                    action='store_true',
                    default=False,
                    help='Remove zero from the MNIST dataset')

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

parser.add_argument('--interpolation-length',
                    action='store',
                    default=10,
                    type=int,
                    help='Specify the sequence length for interpolation')
parser.add_argument('--extrapolation-lengths',
                    action='store',
                    default=[100, 1000],
                    type=ast.literal_eval,
                    help='Specify the sequence lengths used for the extrapolation dataset')

parser.add_argument('--solved-accumulator',
                    action='store_true',
                    default=False,
                    help='Should the accumulator be hard set to either addition or multiplication')
parser.add_argument('--softmax-transform',
                    action='store_true',
                    default=False,
                    help='Should a softmax transformation be used to control the output of the CNN model')
parser.add_argument('--nac-mul',
                    action='store',
                    default='none',
                    choices=['none', 'normal', 'safe', 'max-safe', 'mnac'],
                    type=str,
                    help='Make the second NAC a multiplicative NAC, used in case of a just NAC network.')
parser.add_argument('--nalu-bias',
                    action='store_true',
                    default=False,
                    help='Enables bias in the NALU gate')
parser.add_argument('--nalu-two-nac',
                    action='store_true',
                    default=False,
                    help='Uses two independent NACs in the NALU Layer')
parser.add_argument('--nalu-two-gate',
                    action='store_true',
                    default=False,
                    help='Uses two independent gates in the NALU Layer')
parser.add_argument('--nalu-mul',
                    action='store',
                    default='normal',
                    choices=['normal', 'safe', 'trig', 'max-safe', 'mnac'],
                    help='Multplication unit, can be normal, safe, trig')
parser.add_argument('--nalu-gate',
                    action='store',
                    default='normal',
                    choices=['normal', 'regualized', 'obs-gumbel', 'gumbel'],
                    type=str,
                    help='Can be normal, regualized, obs-gumbel, or gumbel')

parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help=f'Force no CUDA (cuda usage is detected automatically as {torch.cuda.is_available()})')
parser.add_argument('--name-prefix',
                    action='store',
                    default='sequence_mnist',
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
print(f'  - layer_type: {args.layer_type}')
print(f'  - operation: {args.operation}')
print(f'  - regualizer: {args.regualizer}')
print(f'  - remove_zero: {args.remove_zero}')
print(f'  -')
print(f'  - max_epochs: {args.max_epochs}')
print(f'  - batch_size: {args.batch_size}')
print(f'  - seed: {args.seed}')
print(f'  -')
print(f'  - interpolation_length: {args.interpolation_length}')
print(f'  - extrapolation_lengths: {args.extrapolation_lengths}')
print(f'  -')
print(f'  - solved_accumulator: {args.solved_accumulator}')
print(f'  - softmax_transform: {args.softmax_transform}')
print(f'  - nac_mul: {args.nac_mul}')
print(f'  - nalu_bias: {args.nalu_bias}')
print(f'  - nalu_two_nac: {args.nalu_two_nac}')
print(f'  - nalu_two_gate: {args.nalu_two_gate}')
print(f'  - nalu_mul: {args.nalu_mul}')
print(f'  - nalu_gate: {args.nalu_gate}')
print(f'  -')
print(f'  - cuda: {args.cuda}')
print(f'  - name_prefix: {args.name_prefix}')
print(f'  - remove_existing_data: {args.remove_existing_data}')
print(f'  - verbose: {args.verbose}')

# Prepear logging
summary_writer = stable_nalu.writer.SummaryWriter(
    f'{args.name_prefix}/{args.layer_type.lower()}'
    f'{"-nac-" if args.nac_mul != "none" else ""}'
    f'{"n" if args.nac_mul == "normal" else ""}'
    f'{"s" if args.nac_mul == "safe" else ""}'
    f'{"s" if args.nac_mul == "max-safe" else ""}'
    f'{"t" if args.nac_mul == "trig" else ""}'
    f'{"m" if args.nac_mul == "mnac" else ""}'
    f'{"-nalu-" if (args.nalu_bias or args.nalu_two_nac or args.nalu_two_gate or args.nalu_mul != "normal" or args.nalu_gate != "normal") else ""}'
    f'{"b" if args.nalu_bias else ""}'
    f'{"2n" if args.nalu_two_nac else ""}'
    f'{"2g" if args.nalu_two_gate else ""}'
    f'{"s" if args.nalu_mul == "safe" else ""}'
    f'{"s" if args.nalu_mul == "max-safe" else ""}'
    f'{"t" if args.nalu_mul == "trig" else ""}'
    f'{"m" if args.nalu_mul == "mnac" else ""}'
    f'{"r" if args.nalu_gate == "regualized" else ""}'
    f'{"u" if args.nalu_gate == "gumbel" else ""}'
    f'{"uu" if args.nalu_gate == "obs-gumbel" else ""}'
    f'_o-{args.operation.lower()}'
    f'_f-{"r" if args.remove_zero else "k"}'
    f'_m-{"s" if args.softmax_transform else "d"}-{"s" if args.solved_accumulator else "d"}'
    f'_r-{args.regualizer}'
    f'_i-{args.interpolation_length}'
    f'_e-{"-".join(map(str, args.extrapolation_lengths))}'
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
    operation=args.operation,
    use_cuda=args.cuda,
    seed=args.seed,
    remove_zero=args.remove_zero
)
dataset_train = dataset.fork(seq_length=args.interpolation_length, subset='train').dataloader(shuffle=True)
# Seeds are from random.org
dataset_valid_validation = dataset.fork(seq_length=args.interpolation_length, subset='train',
                                        seed=62379872).dataloader(shuffle=False)
dataset_valid_classification = dataset.fork(seq_length=1, subset='test',
                                            seed=47430696).dataloader(shuffle=False)
dataset_valid_extrapolations = [
    ( seq_length,
      dataset.fork(seq_length=seq_length, subset='test',
                   seed=88253339).dataloader(shuffle=False)
    ) for seq_length in args.extrapolation_lengths
]

# setup model
model = stable_nalu.network.SequentialMnistNetwork(
    args.layer_type,
    output_size=dataset.get_item_shape().target[-1],
    sequental_output=dataset.get_item_shape().target[0] is None,
    writer=summary_writer.every(100) if args.verbose else None,
    softmax_transform=args.softmax_transform,
    solved_accumulator=args.solved_accumulator,
    nac_mul=args.nac_mul,
    nalu_bias=args.nalu_bias,
    nalu_two_nac=args.nalu_two_nac,
    nalu_two_gate=args.nalu_two_gate,
    nalu_mul=args.nalu_mul,
    nalu_gate=args.nalu_gate,
)
model.reset_parameters()
if args.cuda:
    model.cuda()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())


def test_mnist(dataloader):
    with torch.no_grad(), model.no_internal_logging(), model.no_random():
        acc_loss = 0
        for x, t in dataloader:
            # forward
            l, _ = model(x)
            acc_loss += criterion(l, t).item() * len(t)

        return acc_loss / len(dataloader.dataset)

def test_model(dataloader):
    with torch.no_grad(), model.no_internal_logging(), model.no_random():
        acc_loss = 0
        for x, t in dataloader:
            # forward
            _, y = model(x)
            acc_loss += criterion(y, t).item() * len(t)

        return acc_loss / len(dataloader.dataset)

# Train model
global_step = 0
for epoch_i in range(0, args.max_epochs + 1):
    for i_train, (x_train, t_train) in enumerate(dataset_train):
        global_step += 1
        summary_writer.set_iteration(global_step)
        summary_writer.add_scalar('epoch', epoch_i)

        # Prepear model
        model.set_parameter('tau', max(0.5, math.exp(-1e-5 * global_step)))
        optimizer.zero_grad()

        # Log validation
        if epoch_i % 100 == 0 and i_train == 0:
            validation_error = test_model(dataset_valid_validation)
            mnist_classification_error = test_mnist(dataset_valid_classification)
            summary_writer.add_scalar('loss/valid/validation', validation_error)
            summary_writer.add_scalar('loss/valid/mnist_classification', mnist_classification_error)

            for seq_length, dataloader in dataset_valid_extrapolations:
                summary_writer.add_scalar(f'loss/valid/extrapolation/{seq_length}', test_model(dataloader))

        # forward
        _, y_train = model(x_train)
        regualizers = model.regualizer()

        loss_train_criterion = criterion(y_train, t_train)
        loss_train_regualizer = args.regualizer * (1 - math.exp(-1e-5 * epoch_i)) * (regualizers['W'] + regualizers['g']) + 1 * regualizers['z'] + 1 * regualizers['W-OOB']
        loss_train = loss_train_criterion + loss_train_regualizer

        # Log loss
        summary_writer.add_scalar('loss/train/critation', loss_train_criterion)
        summary_writer.add_scalar('loss/train/regualizer', loss_train_regualizer)
        summary_writer.add_scalar('loss/train/total', loss_train)
        if epoch_i % 100 == 0 and i_train == 0:
            print('train %d: %.5f, valid: %.5f, mnist: %.5f' % (epoch_i, loss_train_criterion, validation_error, mnist_classification_error))

        # Optimize model
        if loss_train.requires_grad:
            loss_train.backward()
            optimizer.step()
        model.optimize(loss_train_criterion)

        # Log gradients if in verbose mode
        if args.verbose:
            model.log_gradients()

# Write results for this training
print(f'finished:')
print(f'  - loss_train: {loss_train}')
print(f'  - validation: {validation_error}')
print(f'  - mnist_classification: {mnist_classification_error}')

# Use saved weights to visualize the intermediate values.
stable_nalu.writer.save_model(summary_writer.name, model)
