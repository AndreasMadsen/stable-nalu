
import math
import torch
import stable_nalu
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Run the number translation task')
parser.add_argument('--layer-type',
                    action='store',
                    default='NALU',
                    choices=list(stable_nalu.network.NumberTranslationNetwork.UNIT_NAMES),
                    type=str,
                    help='Specify the layer type, e.g. RNN-tanh, LSTM, NAC, NALU')
parser.add_argument('--seed',
                    action='store',
                    default=0,
                    type=int,
                    help='Specify the seed to use')
parser.add_argument('--max-epochs',
                    action='store',
                    default=10000,
                    type=int,
                    help='Specify the max number of epochs to use')
parser.add_argument('--cuda',
                    action='store',
                    default=torch.cuda.is_available(),
                    type=bool,
                    help=f'Should CUDA be used (detected automatically as {torch.cuda.is_available()})')
parser.add_argument('--verbose',
                    action='store_true',
                    default=False,
                    help='Should network measures (e.g. gates) and gradients be shown')
args = parser.parse_args()

# Print configuration
print(f'running')
print(f'  - seed: {args.seed}')
print(f'  - layer_type: {args.layer_type}')
print(f'  - cuda: {args.cuda}')
print(f'  - verbose: {args.verbose}')
print(f'  - max_epochs: {args.max_epochs}')

# Prepear logging
results_writer = stable_nalu.writer.ResultsWriter('number_translation')
summary_writer = stable_nalu.writer.SummaryWriter(
    f'number_translation/{args.layer_type.lower()}_{args.seed}'
)

# Set seed
torch.manual_seed(args.seed)

# Setup datasets
dataset = stable_nalu.dataset.NumberTranslationDataset(
    use_cuda=args.cuda,
    seed=args.seed
)
dataset_train = dataset.fork(subset='train').dataloader()
dataset_valid = dataset.fork(subset='valid').dataloader()

# setup model
model = stable_nalu.network.NumberTranslationNetwork(
    args.layer_type,
    writer=summary_writer.every(150) if args.verbose else None
)
if args.cuda:
    model.cuda()
model.reset_parameters()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

def test_model(dataloader):
    with torch.no_grad(), model.no_internal_logging():
        acc_loss = 0
        for x, t in dataloader:
            # forward
            y = model(x)
            acc_loss += criterion(y, t).item() * len(t)

        return acc_loss / len(dataloader.dataset)

# Train model
global_step = 0
for epoch_i in range(args.max_epochs):
    for i_train, (x_train, t_train) in enumerate(dataset_train):
        global_step += 1
        summary_writer.set_iteration(global_step)

        # Prepear model
        model.set_parameter('tau', max(0.5, math.exp(-1e-5 * global_step)))
        optimizer.zero_grad()

        # Log validation
        if epoch_i % 50 == 0 and i_train == 0:
            summary_writer.add_scalar('loss/valid', test_model(dataset_valid))

        # forward
        y_train = model(x_train)
        loss_train_criterion = criterion(y_train, t_train)
        loss_train_regualizer = 0.1 * (1 - math.exp(-1e-5 * global_step)) * model.regualizer()
        loss_train = loss_train_criterion + loss_train_regualizer

        # Log loss
        summary_writer.add_scalar('loss/train/critation', loss_train_criterion)
        summary_writer.add_scalar('loss/train/regualizer', loss_train_regualizer)
        summary_writer.add_scalar('loss/train/total', loss_train)
        if epoch_i % 50 == 0 and i_train == 0:
            print(f'train {epoch_i}: {loss_train_criterion}')

        # Optimize model
        if loss_train.requires_grad:
            loss_train.backward()
            optimizer.step()
        model.optimize(loss_train_criterion)

        # Log gradients if in verbose mode
        if args.verbose and epoch_i % 10 == 0 and i_train == 0:
            model.log_gradients()

# Compute losses
loss_train = test_model(dataset_train)
loss_valid = test_model(dataset_valid)

# Write results for this training
print(f'finished:')
print(f'  - loss_train: {loss_train}')
print(f'  - loss_valid: {loss_valid}')

# save results
results_writer.add({
    'seed': args.seed,
    'layer_type': args.layer_type,
    'cuda': args.cuda,
    'verbose': args.verbose,
    'max_epochs': args.max_epochs,
    'loss_train': loss_train,
    'loss_valid': loss_valid,
})
