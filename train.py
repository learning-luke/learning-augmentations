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
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from tensorboardX import SummaryWriter
from utils.storage import build_experiment_folder, save_statistics, save_checkpoint, restore_model, save_image_batch
from models.model_selector import ModelSelector
from utils.datasets import load_dataset
from utils.administration import get_start_epoch, get_best_epoch, parse_args

######################################################################################################### Args
args = parse_args()

debug = False
if debug:
    args.logit_loss = 0
    args.learn_cutout = 1
    # args.use_fc = 1
    args.upsample_type = 'nearest'

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
                     "train_loss_masked",
                     "test_loss",
                     "test_loss_masked",
                     "train_acc",
                     "train_acc_masked",
                     "test_acc",
                     "test_acc_masked"
                     ],
                    create=True)

######################################################################################################### Model

net = ModelSelector(dataset=args.dataset,
                    in_shape=in_shape,
                    filters=args.filters,
                    activation=args.activation,
                    widen_factor=args.widen_factor,
                    num_classes=10,
                    resdepth=args.resdepth).select(args.model, args.use_fc, args.upsample_type)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    
######################################################################################################### Optimisation

def logit_error(logit_x, logit_y, targets):
    sm = nn.Softmax(dim=1)
    probs_x = sm(logit_x)
    probs_y = sm(logit_y)
    mask = torch.ones_like(logit_x)
    for i, t in enumerate(targets):
        mask[i, t] = 0
        mask[i, t] = 0
    return torch.mean(torch.abs(probs_x - probs_y)[mask==1])



def mean_abs_error(x, y):
    return torch.mean(torch.abs(x-y))

def rms_error(x, y):
    return torch.mean((x - y)**2)

criterion = nn.CrossEntropyLoss()
criterion_sim = mean_abs_error
criterion_reg = nn.SmoothL1Loss()
BCE_stable = torch.nn.BCEWithLogitsLoss()
if args.logit_loss:
    criterion_sim = logit_error
optimizer_c = optim.SGD([p for name, p in net.named_parameters() if 'path' not in name], lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
optimizer_g = optim.SGD([p for name, p in net.named_parameters() if 'path' in name], lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)#optim.Adam([p for name, p in net.named_parameters() if 'path' in name], lr=0.001, amsgrad=True)

if args.scheduler == 'CosineAnnealing':
    scheduler_c = CosineAnnealingLR(optimizer=optimizer_c, T_max=args.max_epochs, eta_min=0)
    scheduler_g = CosineAnnealingLR(optimizer=optimizer_g, T_max=args.max_epochs, eta_min=0)
else:
    scheduler_c = MultiStepLR(optimizer_c, milestones=args.milestones, gamma=0.2)
    scheduler_g = MultiStepLR(optimizer_g, milestones=args.milestones, gamma=0.2)

######################################################################################################### Restoring/logging

restore_model(net, optimizer_c, optimizer_g, args)
writer = SummaryWriter(log_dir=logs_filepath)


def onehot(batch, depth=10):
    ones = torch.eye(depth).to(device)
    return ones.index_select(0, batch)
######################################################################################################### Training

def get_secondary_targets(logits, targets):
    new_targets = torch.zeros_like(targets).to(device)
    sorted, indices = torch.sort(logits, dim=1)
    for i in range(targets.size(0)):
        if indices[i][-1] != targets[i]:
            new_targets[i] = indices[i][-1]
        else:
            new_targets[i] = indices[i][-2]
    return new_targets

def get_loss(inputs, targets):

    all_logits, before_paths, all_h, choice = net(inputs, use_input=True)
    logits = all_logits[0]
    logits_masked_input = all_logits[1]

    loss = criterion(logits, targets)
    loss_masked_input = criterion(logits_masked_input, targets)

    return logits, logits_masked_input, loss, loss_masked_input, choice



def train(epoch):
    global net
    net.train()
    train_loss = 0
    train_loss_masked = 0
    correct = 0
    correct_masked = 0
    total = 0
    with tqdm.tqdm(initial=0, total=n_train_batches) as train_pbar:
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_c.zero_grad()

            logits, logits_masked_input, loss, loss_masked_input, choice = get_loss(inputs, targets)

            train_loss += loss.item()
            train_loss_masked += loss_masked_input.item()

            loss.backward(retain_graph=True)
            optimizer_c.step()

            optimizer_g.zero_grad()
            (loss_masked_input - loss).backward()
            optimizer_g.step()



            total += targets.size(0)
            correct += logits.max(1)[1].eq(targets).sum().item()
            correct_masked += logits_masked_input.max(1)[1].eq(targets).sum().item()

            current_lr = optimizer_c.param_groups[0]['lr']
            iter_out = 'Training {};, LR: {:0.5f}, Loss: {:0.4f}, Loss_m: {:0.4f}, Acc: {:0.4f}, Acc_m: {:0.4f} '.format(
                batch_idx,
                current_lr,
                train_loss / (batch_idx + 1),
                train_loss_masked / (batch_idx + 1),
                100. * correct / total,
                100. * correct_masked / total,
            )

            train_pbar.set_description(iter_out)
            train_pbar.update()



            if batch_idx == 0:
                ordering = np.argsort(targets.detach().cpu().numpy())
                save_image_batch("{}/train/{}_inputs.png".format(images_filepath, epoch), inputs[ordering])
                save_image_batch("{}/train/{}_mask.png".format(images_filepath, epoch), choice[:, 0:1, :, :][ordering])
                save_image_batch("{}/train/{}_inputs_masked.png".format(images_filepath, epoch), (inputs*choice[:, 0:1, :, :])[ordering])


    return train_loss / n_train_batches, train_loss_masked / n_train_batches, correct / total, correct_masked / total

def test(epoch):
    global best_acc, net
    net.eval()
    test_loss = 0
    test_loss_masked = 0
    correct = 0
    correct_masked = 0
    total = 0

    with tqdm.tqdm(initial=0, total=n_test_batches) as test_pbar:
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            logits, logits_masked_input, loss, loss_masked_input, choice = get_loss(inputs, targets)

            test_loss += loss.item()
            test_loss_masked += loss_masked_input.item()


            total += targets.size(0)
            correct += logits.max(1)[1].eq(targets).sum().item()
            correct_masked += logits_masked_input.max(1)[1].eq(targets).sum().item()

            iter_out = 'Testing; Loss: {:0.4f}, Loss_m: {:0.4f}, Acc: {:0.4f}, Acc_m: {:0.4f}'.format(
                test_loss / (batch_idx + 1),
                test_loss_masked / (batch_idx + 1),
                100. * correct / total,
                100. * correct_masked / total
            )

            test_pbar.set_description(iter_out)
            test_pbar.update()

            if batch_idx == 0:
                ordering = np.argsort(targets.detach().cpu().numpy())
                save_image_batch("{}/test/{}_inputs.png".format(images_filepath, epoch), inputs[ordering])
                save_image_batch("{}/train/{}_inputs.png".format(images_filepath, epoch), inputs[ordering])
                save_image_batch("{}/train/{}_mask.png".format(images_filepath, epoch), choice[:, 0:1, :, :][ordering])
                save_image_batch("{}/train/{}_inputs_masked.png".format(images_filepath, epoch),(inputs * choice[:, 0:1, :, :])[ordering])


    return test_loss / n_test_batches, test_loss_masked / n_test_batches, correct / total, correct_masked / total

if __name__ == "__main__":
    with tqdm.tqdm(initial=start_epoch, total=args.max_epochs) as epoch_pbar:
        for epoch in range(start_epoch, args.max_epochs):
            scheduler_c.step(epoch=epoch)
            scheduler_g.step(epoch=epoch)

            train_loss, train_loss_masked, train_acc, train_acc_masked = train(epoch)
            test_loss, test_loss_masked, test_acc, test_acc_masked = test(epoch)

            save_statistics(logs_filepath, "result_summary_statistics",
                            [epoch,
                             train_loss,
                             train_loss_masked,
                             test_loss,
                             test_loss_masked,
                             train_acc,
                             train_acc_masked,
                             test_acc,
                             test_acc_masked])

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
                'optimizer_c': optimizer_c.state_dict(),
                'optimizer_g': optimizer_g.state_dict(),
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

