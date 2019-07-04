
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
                        'cumsum', 'sum', 'cumprod', 'prod'
                    ],
                    type=str,
                    help='Specify the operation to use, sum or count')
parser.add_argument('--regualizer',
                    action='store',
                    default=0.1,
                    type=float,
                    help='Specify the regualization lambda to be used')
parser.add_argument('--regualizer-z',
                    action='store',
                    default=0,
                    type=float,
                    help='Specify the z-regualization lambda to be used')
parser.add_argument('--regualizer-oob',
                    action='store',
                    default=1,
                    type=float,
                    help='Specify the oob-regualization lambda to be used')
parser.add_argument('--mnist-digits',
                    action='store',
                    default=[0,1,2,3,4,5,6,7,8,9],
                    type=lambda str: list(map(int,str)),
                    help='MNIST digits to use')
parser.add_argument('--mnist-outputs',
                    action='store',
                    default=1,
                    type=int,
                    help='number of MNIST to use, more than 1 adds redundant values')
parser.add_argument('--model-simplification',
                    action='store',
                    default='none',
                    choices=[
                        'none', 'solved-accumulator', 'pass-through'
                    ],
                    type=str,
                    help='Simplifiations applied to the model')

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
parser.add_argument('--nac-oob',
                    action='store',
                    default='clip',
                    choices=['regualized', 'clip'],
                    type=str,
                    help='Choose of out-of-bound should be handled by clipping or regualization.')
parser.add_argument('--regualizer-shape',
                    action='store',
                    default='linear',
                    choices=['squared', 'linear'],
                    type=str,
                    help='Use either a squared or linear shape for the bias and oob regualizer.')
parser.add_argument('--mnac-epsilon',
                    action='store',
                    default=0,
                    type=float,
                    help='Set the idendity epsilon for MNAC.')
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
print(f'  - regualizer_z: {args.regualizer_z}')
print(f'  - regualizer_oob: {args.regualizer_oob}')
print(f'  - mnist_digits: {args.mnist_digits}')
print(f'  - mnist_outputs: {args.mnist_outputs}')
print(f'  - model_simplification: {args.model_simplification}')
print(f'  -')
print(f'  - max_epochs: {args.max_epochs}')
print(f'  - batch_size: {args.batch_size}')
print(f'  - seed: {args.seed}')
print(f'  -')
print(f'  - interpolation_length: {args.interpolation_length}')
print(f'  - extrapolation_lengths: {args.extrapolation_lengths}')
print(f'  -')
print(f'  - softmax_transform: {args.softmax_transform}')
print(f'  - nac_mul: {args.nac_mul}')
print(f'  - nac_oob: {args.nac_oob}')
print(f'  - regualizer_shape: {args.regualizer_shape}')
print(f'  - mnac_epsilon: {args.mnac_epsilon}')
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
    f'_d-{"".join(map(str, args.mnist_digits))}'
    f'_h-{args.mnist_outputs}'
    f'_op-{args.operation.lower()}'
    f'_oob-{"c" if args.nac_oob == "clip" else "r"}'
    f'_rs-{args.regualizer_shape}'
    f'_eps-{args.mnac_epsilon}'
    f'_r-{args.regualizer}-{args.regualizer_z}-{args.regualizer_oob}'
    f'_m-{"s" if args.softmax_transform else "l"}-{args.model_simplification[0]}'
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
    mnist_digits=args.mnist_digits
)
dataset_train = dataset.fork(seq_length=args.interpolation_length, subset='train').dataloader(shuffle=True)
# Seeds are from random.org
dataset_train_validation = dataset.fork(seq_length=args.interpolation_length, subset='train',
                                        seed=62379872).dataloader(shuffle=False)
dataset_train_classification = dataset.fork(seq_length=1, subset='train',
                                           seed=3383872).dataloader(shuffle=False)
dataset_test_classification = dataset.fork(seq_length=1, subset='test',
                                           seed=47430696).dataloader(shuffle=False)
dataset_test_extrapolations = [
    ( seq_length,
      dataset.fork(seq_length=seq_length, subset='test',
                   seed=88253339).dataloader(shuffle=False)
    ) for seq_length in args.extrapolation_lengths
]

# setup model
model = stable_nalu.network.SequentialMnistNetwork(
    args.layer_type,
    output_size=dataset.get_item_shape().target[-1],
    writer=summary_writer.every(100).verbose(args.verbose),
    mnist_digits=args.mnist_digits,
    mnist_outputs=args.mnist_outputs,
    model_simplification=args.model_simplification,
    softmax_transform=args.softmax_transform,
    nac_mul=args.nac_mul,
    nac_oob=args.nac_oob,
    regualizer_shape=args.regualizer_shape,
    regualizer_z=args.regualizer_z,
    mnac_epsilon=args.mnac_epsilon,
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

seq_index = slice(None) if dataset.get_item_shape().target[0] is None else -1

def accuracy(y, t):
    return torch.mean((torch.round(y) == t).float())

def test_mnist(dataloader):
    with torch.no_grad(), model.no_internal_logging(), model.no_random():
        mse_loss = 0
        acc_last = 0
        for x, t in dataloader:
            # forward
            l, _ = model(x)
            mse_loss += min(criterion(l[:,0,i], t[:,0,0]).item() for i in range(l.size(-1))) * len(t)
            acc_last += max(accuracy(l[:,0,i], t[:,0,0]).item() for i in range(l.size(-1))) * len(t)

        return [
            mse_loss / len(dataloader.dataset),
            acc_last / len(dataloader.dataset)
        ]

def test_model(dataloader):
    with torch.no_grad(), model.no_internal_logging(), model.no_random():
        mse_loss = 0
        acc_all = 0
        acc_last = 0
        for x, t in dataloader:
            # forward
            _, y = model(x)
            mse_loss += criterion(y[:,seq_index,:], t[:,seq_index,:]).item() * len(t)
            acc_all += accuracy(y[:,seq_index,:], t[:,seq_index,:]).item() * len(t)
            acc_last += accuracy(y[:,-1,:], t[:,-1,:]).item() * len(t)

        return [
            mse_loss / len(dataloader.dataset),
            acc_all / len(dataloader.dataset),
            acc_last / len(dataloader.dataset)
        ]

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
        if epoch_i % 5 == 0 and i_train == 0:
            (model_train_validation_mse,
             model_train_validation_acc_all,
             model_train_validation_acc_last) = test_model(dataset_train_validation)
            mnist_train_classification_mse, mnist_train_classification_acc = test_mnist(dataset_train_classification)
            mnist_test_classification_mse, mnist_test_classification_acc = test_mnist(dataset_test_classification)

            summary_writer.add_scalar('loss/valid/validation/mse', model_train_validation_mse)
            summary_writer.add_scalar('loss/valid/validation/acc/all', model_train_validation_acc_all)
            summary_writer.add_scalar('loss/valid/validation/acc/last', model_train_validation_acc_last)
            summary_writer.add_scalar('loss/valid/mnist/mse', mnist_train_classification_mse)
            summary_writer.add_scalar('loss/valid/mnist/acc', mnist_train_classification_acc)
            summary_writer.add_scalar('loss/test/mnist/mse', mnist_test_classification_mse)
            summary_writer.add_scalar('loss/test/mnist/acc', mnist_test_classification_acc)

            for seq_length, dataloader in dataset_test_extrapolations:
                (model_test_extrapolation_mse,
                 model_test_extrapolation_acc_all,
                 model_test_extrapolation_acc_last) = test_model(dataloader)
                summary_writer.add_scalar(f'loss/test/extrapolation/{seq_length}/mse', model_test_extrapolation_mse)
                summary_writer.add_scalar(f'loss/test/extrapolation/{seq_length}/acc/all', model_test_extrapolation_acc_all)
                summary_writer.add_scalar(f'loss/test/extrapolation/{seq_length}/acc/last', model_test_extrapolation_acc_last)

        # forward
        with summary_writer.force_logging(epoch_i % 5 == 0 and i_train == 0):
            mnist_y_train, y_train = model(x_train)
        regualizers = model.regualizer()

        loss_train_criterion = criterion(y_train[:,seq_index,:], t_train[:,seq_index,:])
        loss_train_regualizer = args.regualizer * (1 - math.exp(-1e-5 * global_step)) * (regualizers['W'] + regualizers['g']) + args.regualizer_z * regualizers['z'] + args.regualizer_oob * regualizers['W-OOB']
        loss_train = loss_train_criterion + loss_train_regualizer

        # Log loss
        summary_writer.add_scalar('loss/train/accuracy/all', accuracy(y_train[:,seq_index,:], t_train[:,seq_index,:]))
        summary_writer.add_scalar('loss/train/accuracy/last', accuracy(y_train[:,-1,:], t_train[:,-1,:]))
        summary_writer.add_scalar('loss/train/critation', loss_train_criterion)
        summary_writer.add_scalar('loss/train/regualizer', loss_train_regualizer)
        summary_writer.add_scalar('loss/train/total', loss_train)
        if epoch_i % 5 == 0 and i_train == 0:
            summary_writer.add_tensor('MNIST/train',
                                      torch.cat([mnist_y_train[:,0,:], t_train[:,0,:]], dim=1))
            print('train %d: %.5f, valid: %.5f, %.3f (acc[last]), mnist: %.5f, %.3f (acc)' % (
                epoch_i, loss_train_criterion,
                model_train_validation_mse, model_train_validation_acc_last,
                mnist_test_classification_mse, mnist_test_classification_acc
            ))

        # Optimize model
        if loss_train.requires_grad:
            loss_train.backward()
            optimizer.step()
        model.optimize(loss_train_criterion)

        # Log gradients if in verbose mode
        with summary_writer.force_logging(epoch_i % 5 == 0 and i_train == 0):
            model.log_gradients()

# Write results for this training
print(f'finished:')
print(f'  - loss_train: {loss_train}')
print(f'  - validation: {validation_error}')
print(f'  - mnist_classification: {mnist_classification_error}')

# Use saved weights to visualize the intermediate values.
stable_nalu.writer.save_model(summary_writer.name, model)
