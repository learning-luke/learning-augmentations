'''
Adapted from: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py,
and my own work
'''
import os

import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from tensorboardX import SummaryWriter
from utils.storage import build_experiment_folder, save_statistics, save_checkpoint, restore_model, save_image_batch
from models.model_selector import ModelSelector
from utils.datasets import load_dataset
from utils.administration import get_start_epoch, get_best_epoch, parse_args

######################################################################################################### Args
args = parse_args()

debug = True
if debug:
    pass


######################################################################################################### Data
trainloader, testloader, in_shape = load_dataset(args)

n_train_batches = len(trainloader)
n_train_images = len(trainloader.dataset)
n_test_batches = len(testloader)
n_test_images = len(testloader.dataset)


assert args.train_log_interval < n_train_batches, \
    'Log interval ({}) must be smaller than train batches ({})'.format(args.train_log_interval, n_train_batches)

print("Data loaded successfully ")
print("Training --> {} images and {} batches".format(n_train_images, n_train_batches))
print("Testing --> {} images and {} batches".format(n_test_images, n_test_batches))

######################################################################################################### Admin
torch.manual_seed(args.seed)
rng = np.random.RandomState(seed=args.seed)  # set seed
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.device = device
saved_models_filepath, logs_filepath, images_filepath = build_experiment_folder(args)

start_epoch, latest_loadpath = get_start_epoch(args)
args.latest_loadpath = latest_loadpath
best_epoch, best_test_acc = get_best_epoch(args)
if best_epoch >= 0:
    print('Best evaluation acc so far at {} epochs: {:0.2f}'.format(best_epoch, best_test_acc))

if not args.resume:
    save_statistics(logs_filepath, "result_summary_statistics",
                    ["epoch",
                     "train_loss",
                     "test_loss",
                     "train_loss_sim",
                     "test_loss_sim",
                     "train_acc",
                     "test_acc",
                     ],
                    create=True)

######################################################################################################### Model

net = ModelSelector(dataset=args.dataset,
                    in_shape=in_shape,
                    filters=args.filters,
                    activation=args.activation,
                    widen_factor=args.widen_factor,
                    num_classes=10,
                    resdepth=args.resdepth).select(args.model)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    
######################################################################################################### Optimisation

def rms_error(x, y):
    return torch.mean((x - y)**2)

criterion = nn.CrossEntropyLoss()
criterion_sim = rms_error
optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)

if args.scheduler == 'CosineAnnealing':
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.max_epochs, eta_min=0)
else:
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=0.2)

######################################################################################################### Restoring/logging

restore_model(net, optimizer, args)
writer = SummaryWriter(log_dir=logs_filepath)

######################################################################################################### Training

def train(epoch):
    global net
    net.train()
    train_loss = 0
    train_loss_sim = 0
    correct = 0
    total = 0
    with tqdm.tqdm(initial=0, total=n_train_batches) as train_pbar:
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            all_logits, before_paths = net(inputs)
            loss = 0
            loss_sim = 0
            num_comparisons = 0
            logits = torch.zeros_like(all_logits[0])
            for pathi in range(args.num_paths):
                loss += criterion(all_logits[pathi], targets)
                logits += all_logits[pathi]
                for other_pathi in range(pathi+1, args.num_paths):
                    loss_sim += criterion_sim(before_paths[pathi], before_paths[other_pathi])
                    num_comparisons += 1

            logits /= args.num_paths
            loss_sim = (loss_sim/num_comparisons) * args.sim_loss_mult
            loss += loss_sim


            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loss_sim += loss_sim.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            current_lr = optimizer.param_groups[0]['lr']
            iter_out = 'Training {};, LR: {:0.5f}, Loss: {:0.4f}, Loss_sim: {:0.4f}, Acc: {:0.4f};'.format(
                batch_idx,
                current_lr,
                train_loss / (batch_idx + 1),
                train_loss_sim / (batch_idx + 1),
                100. * correct / total,
            )

            train_pbar.set_description(iter_out)
            train_pbar.update()

            # tensorboard logs
            if (batch_idx+1) % args.train_log_interval == 0:
                steps = epoch*n_train_batches + (batch_idx+1)
                names = ['learning_rate', 'loss', 'acc']
                vars = [current_lr, train_loss / (batch_idx + 1), 100. * correct / total]
                for n, v in zip(names, vars):
                    writer.add_scalar('train/'+n, v, steps)

            if batch_idx == 0:
                save_image_batch("{}/train/{}_inputs.png".format(images_filepath, epoch), inputs)
                for pathi in range(args.num_paths):
                    outputs = before_paths[:,3*pathi:3*(pathi+1),:,:]
                    save_image_batch("{}/train/{}_outputs_{}.png".format(images_filepath, epoch, pathi), inputs)



    return train_loss / n_train_batches, train_loss_sim / n_train_batches, correct / total

def test(epoch):
    global best_acc, net
    net.eval()
    test_loss = 0
    test_loss_sim = 0
    correct = 0
    total = 0

    with tqdm.tqdm(initial=0, total=n_test_batches) as test_pbar:
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            all_logits, before_paths = net(inputs)
            loss = 0
            loss_sim = 0
            num_comparisons = 0
            logits = torch.zeros_like(all_logits[0])
            for pathi in range(args.num_paths):
                loss += criterion(all_logits[pathi], targets)
                logits += all_logits[pathi]
                for other_pathi in range(pathi + 1, args.num_paths):
                    loss_sim += criterion_sim(before_paths[pathi], before_paths[other_pathi])
                    num_comparisons += 1

            logits /= args.num_paths
            loss_sim = (loss_sim / num_comparisons) * args.sim_loss_mult
            loss += loss_sim
            test_loss_sim += loss_sim.item()
            test_loss += loss.item()

            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            iter_out = 'Testing; Loss: {:0.4f}, Loss_sim: {:0.4f}, Acc: {:0.4f}'.format(
                test_loss / (batch_idx + 1),
                test_loss_sim / (batch_idx + 1),
                100. * correct / total
            )

            test_pbar.set_description(iter_out)
            test_pbar.update()

            if batch_idx == 0:
                save_image_batch("{}/test/{}_inputs.png".format(images_filepath, epoch), inputs)
                for pathi in range(args.num_paths):
                    outputs = before_paths[:,3*pathi:3*(pathi+1),:,:]
                    save_image_batch("{}/test/{}_outputs_{}.png".format(images_filepath, epoch, pathi), inputs)

        # tensorboard logs (once per epoch)
        steps= epoch*n_train_batches
        names = ['loss', 'acc']
        vars = [test_loss / (batch_idx + 1), 100. * correct / total]
        for n, v in zip(names, vars):
            writer.add_scalar('test/' + n, v, steps)

    return test_loss / n_test_batches, test_loss_sim / n_test_batches, correct / total

if __name__ == "__main__":
    with tqdm.tqdm(initial=start_epoch, total=args.max_epochs) as epoch_pbar:
        for epoch in range(start_epoch, start_epoch + args.max_epochs):
            scheduler.step(epoch=epoch)

            train_loss, train_loss_sim, train_acc = train(epoch)
            test_loss, test_loss_sim, test_acc = test(epoch)

            save_statistics(logs_filepath, "result_summary_statistics",
                            [epoch,
                             train_loss,
                             train_loss_sim,
                             test_loss,
                             test_loss_sim,
                             train_acc,
                             test_acc])

            # Saving models
            is_best = False
            previous_best_epoch = -1
            if best_test_acc <= test_acc:
                previous_best_epoch = best_epoch
                best_test_acc = test_acc
                best_epoch = epoch
                is_best = True
            state = {
                'epoch': epoch,
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            epoch_pbar.set_description('Saving at {}/{}_checkpoint.pth.tar'.format(saved_models_filepath, epoch))
            filename = '{}_checkpoint.pth.tar'.format(epoch)

            previous_save = '{}/{}_checkpoint.pth.tar'.format(saved_models_filepath, epoch - 1)
            if os.path.isfile(previous_save):
                os.remove(previous_save)

            previous_best_save = '{}/best_{}_checkpoint.pth.tar'.format(saved_models_filepath, previous_best_epoch)
            if os.path.isfile(previous_best_save) and is_best:
                os.remove(previous_best_save)

            save_checkpoint(state=state,
                            directory=saved_models_filepath,
                            filename=filename,
                            is_best=is_best)
            epoch_pbar.set_description('')
            epoch_pbar.update(1)

