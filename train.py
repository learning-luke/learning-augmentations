'''
Adapted from: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py,
and my own work
'''
import os
import torch.nn.functional as F
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
                     "train_loss_sim",
                     "train_loss_reg",
                     "test_loss",
                     "test_loss_sim",
                     "test_loss_reg",
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
optimizer_classifier = optim.SGD([p for name, p in net.named_parameters() if 'path' not in name], lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
optimizer_generator = optim.Adam([p for name, p in net.named_parameters() if 'path' in name], lr=0.001)#optim.SGD([p for name, p in net.named_parameters() if 'path' in name], lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)

if args.scheduler == 'CosineAnnealing':
    scheduler_classifier = CosineAnnealingLR(optimizer=optimizer_classifier, T_max=args.max_epochs, eta_min=0)
    scheduler_generator = CosineAnnealingLR(optimizer=optimizer_generator, T_max=args.max_epochs, eta_min=0)
else:
    scheduler_classifier = MultiStepLR(optimizer_classifier, milestones=args.milestones, gamma=0.2)
    scheduler_generator = MultiStepLR(optimizer_generator, milestones=args.milestones, gamma=0.2)

######################################################################################################### Restoring/logging

restore_model(net, optimizer_classifier, args)
writer = SummaryWriter(log_dir=logs_filepath)


def onehot(batch, depth=10):
    ones = torch.eye(depth).to(device)
    return ones.index_select(0, batch)
######################################################################################################### Training

def mixed_cross_entropy_loss(logits, targets1, targets2, alpha):
    # Check - are they one hot encoded?

    targets = onehot(targets1).float() * alpha + onehot(targets2).float() * (1-alpha)

    logsoftmax = F.log_softmax(logits, dim=1).float()

    cross_entropy = -torch.mean(torch.sum(targets * logsoftmax, dim=1))
    return cross_entropy

def get_loss2(inputs, targets, alpha_low=0.7, alpha_high=1.0):
    inputs1 = inputs[:inputs.size(0)//2]
    inputs2 = inputs[inputs.size(0)//2:]
    targets1 = targets[:targets.size(0)//2]
    targets2 = targets[targets.size(0)//2:]
    alpha = torch.empty(inputs1.size(0), 1).uniform_(alpha_low, alpha_high).to(device)

    all_logits, x_up, all_h = net(inputs1, inputs2, alpha)
    logits_mixed = all_logits[0]
    logits1 = all_logits[1]
    logits2 = all_logits[2]

    # loss = mixed_cross_entropy_loss(logits_mixed, targets1, targets2, alpha)
    loss_c = criterion(logits1, targets1)
    loss_c += criterion(logits2, targets2)

    loss_g = rms_error(logits_mixed, logits1 * alpha + logits2 * (1-alpha)) * args.regularise_mult
    for hi in range(0, 3):
        loss_g += rms_error(all_h[hi].detach(), all_h[hi+3].detach() * alpha.view(alpha.size(0),1,1,1) + all_h[hi+6].detach() * (1-alpha).view(alpha.size(0),1,1,1))
    # y_pred_fake = logits_mixed
    # y_pred = logits1 * alpha + logits2 * (1-alpha)
    # loss_g = (torch.mean(torch.nn.ReLU()(1.0 + (y_pred - torch.mean(y_pred_fake)))) + torch.mean(torch.nn.ReLU()(1.0 - (y_pred_fake - torch.mean(y_pred)))))/2
    # loss_g = rms_error(logits_mixed, logits2 * (1-alpha)) * args.regularise_mult
    return logits1, loss_c,loss_g, x_up

def get_loss(inputs, targets):
    loss = 0
    loss_sim = 0
    loss_reg = 0
    num_comparisons = 0

    if args.learn_cutout:
        all_logits, before_paths = net(inputs, use_input=False)


        logits = torch.zeros_like(all_logits[0])
        # TODO: learn to combine images?

        loss_reg += criterion_reg(before_paths[0] + before_paths[1], inputs)
        loss_sim += criterion_sim(before_paths[0], before_paths[1])
        loss += criterion(all_logits[1], targets)
        logits += all_logits[1]
        num_comparisons += 1
        loss += criterion(all_logits[2], targets)
        logits += all_logits[2]
        logits /= 2
    else:
        all_logits, before_paths = net(inputs, use_input=True)

        logits = torch.zeros_like(all_logits[0])
        logits += all_logits[0]
        loss += criterion(all_logits[0], targets)
        for pathi in range(1, args.num_paths + 1):

            if args.classify_augmentations:
                loss += criterion(all_logits[pathi], targets)
                logits += all_logits[pathi]

            if args.logit_loss:
                loss_reg += torch.mean((all_logits[0] - all_logits[pathi]) ** 2)

            else:
                loss_reg += criterion_reg(before_paths[pathi - 1], inputs)

            for other_pathi in range(pathi + 1, args.num_paths + 1):
                if args.logit_loss:
                    loss_sim += criterion_sim(all_logits[pathi], all_logits[other_pathi], targets)
                else:
                    loss_sim += criterion_sim(before_paths[pathi - 1], before_paths[other_pathi - 1])
                num_comparisons += 1

        if args.classify_augmentations:
            logits /= (args.num_paths + 1)

    return before_paths, logits, loss, loss_sim, loss_reg, num_comparisons

def train(epoch):
    global net
    net.train()
    train_loss = 0
    train_loss_sim = 0
    train_loss_reg = 0
    correct = 0
    total = 0
    with tqdm.tqdm(initial=0, total=n_train_batches) as train_pbar:
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_classifier.zero_grad()
            optimizer_generator.zero_grad()

            # before_paths, logits, loss, loss_sim, loss_reg, num_comparisons = get_loss(inputs, targets)
            logits, loss_c, loss_g, x_up = get_loss2(inputs, targets, alpha_low=args.alpha_low, alpha_high=args.alpha_high)

            loss_c.backward(retain_graph=True)
            optimizer_classifier.step()
            loss_g.backward()
            optimizer_generator.step()

            train_loss += loss_c.item()
            train_loss_reg += loss_g.item()
            # loss_sim = (loss_sim/num_comparisons) * args.sim_loss_mult
            # loss += loss_sim
            # if args.regularise_mult != 0:
            #     loss += loss_reg
            #     train_loss_reg += loss_reg.item()





            # train_loss_sim += loss_sim.item()
            _, predicted = logits.max(1)
            total += targets[:targets.size(0)//2].size(0)
            correct += predicted.eq(targets[:targets.size(0)//2]).sum().item()

            current_lr = optimizer_classifier.param_groups[0]['lr']
            iter_out = 'Training {};, LR: {:0.5f}, Loss: {:0.4f}, Loss_sim: {:0.4f}, Loss_reg: {:0.4f}, Acc: {:0.4f};'.format(
                batch_idx,
                current_lr,
                train_loss / (batch_idx + 1),
                train_loss_sim / (batch_idx + 1),
                train_loss_reg / (batch_idx + 1),
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
                outputs = torch.cat((x_up, x_up), dim=0)
                save_image_batch("{}/train/{}_outputs.png".format(images_filepath, epoch), outputs)



    return train_loss / n_train_batches, train_loss_sim / n_train_batches, train_loss_reg / n_train_batches, correct / total

def test(epoch):
    global best_acc, net
    net.eval()
    test_loss = 0
    test_loss_sim = 0
    test_loss_reg = 0
    correct = 0
    total = 0

    with tqdm.tqdm(initial=0, total=n_test_batches) as test_pbar:
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # before_paths, logits, loss, loss_sim, loss_reg, num_comparisons = get_loss(inputs, targets)
            logits, loss_c, loss_g, x_up = get_loss2(inputs, targets, alpha_low=0.99, alpha_high=1)
            test_loss_reg += loss_g.item()
            targets = targets
            test_loss += loss_c.item()
            # if args.classify_augmentations:
            #     logits /= (args.num_paths + 1)
            # loss_sim = (loss_sim / num_comparisons) * args.sim_loss_mult
            # loss += loss_sim
            # test_loss_sim += loss_sim.item()
            # if args.regularise_mult != 0:
            #     test_loss_reg += loss_reg.item()

            _, predicted = logits.max(1)
            total += targets[:targets.size(0)//2].size(0)
            correct += predicted.eq(targets[:targets.size(0)//2]).sum().item()

            iter_out = 'Testing; Loss: {:0.4f}, Loss_sim: {:0.4f}, Loss_reg: {:0.4f}, Acc: {:0.4f}'.format(
                test_loss / (batch_idx + 1),
                test_loss_sim / (batch_idx + 1),
                test_loss_reg / (batch_idx + 1),
                100. * correct / total
            )

            test_pbar.set_description(iter_out)
            test_pbar.update()

            if batch_idx == 0:

                save_image_batch("{}/test/{}_inputs.png".format(images_filepath, epoch), inputs)
                outputs = torch.cat((x_up, x_up), dim=0)
                save_image_batch("{}/test/{}_outputs.png".format(images_filepath, epoch), outputs)

        # tensorboard logs (once per epoch)
        steps= epoch*n_train_batches
        names = ['loss', 'acc']
        vars = [test_loss / (batch_idx + 1), 100. * correct / total]
        for n, v in zip(names, vars):
            writer.add_scalar('test/' + n, v, steps)

    return test_loss / n_test_batches, test_loss_sim / n_test_batches, test_loss_reg / n_test_batches, correct / total

if __name__ == "__main__":
    with tqdm.tqdm(initial=start_epoch, total=args.max_epochs) as epoch_pbar:
        for epoch in range(start_epoch, args.max_epochs):
            scheduler_classifier.step(epoch=epoch)
            scheduler_generator.step(epoch=epoch)

            train_loss, train_loss_sim, train_loss_reg, train_acc = train(epoch)
            test_loss, test_loss_sim, test_loss_reg, test_acc = test(epoch)

            save_statistics(logs_filepath, "result_summary_statistics",
                            [epoch,
                             train_loss,
                             train_loss_sim,
                             train_loss_reg,
                             test_loss,
                             test_loss_sim,
                             test_loss_reg,
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
                'optimizer': optimizer_classifier.state_dict(),
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

