from utils.storage import load_statistics
import argparse
import json
import numpy as np

def get_start_epoch(args):
    latest_loadpath = ''
    latest_trained_epoch = -1
    if args.resume:
        from glob import glob
        # search for the latest
        saved_ckpts = glob('{}/*checkpoint.pth.tar'.format(args.saved_models_filepath))
        if len(saved_ckpts) == 0:
            args.resume = 0
        else:
            for fn in saved_ckpts:
                if 'best' not in fn.split('/')[-1]:
                    query_trained_epoch = int(fn.split('/')[-1].split('_')[0])
                    if query_trained_epoch > latest_trained_epoch:
                        latest_trained_epoch = query_trained_epoch
                        latest_loadpath = fn
    start_epoch = 0
    if args.resume:
        start_epoch = latest_trained_epoch + 1

    return start_epoch, latest_loadpath

def get_best_epoch(args):
    best_epoch = -1
    # Calculate the best loss from the results statistics if restarting
    best_test_acc = 0.0
    if args.resume:
        print("Checking {}/{}.csv".format(args.logs_filepath, "result_summary_statistics"))
        results = load_statistics(args.logs_filepath, "result_summary_statistics")
        maxi = np.argmax(results['test_acc'])
        best_test_acc = results['test_acc'][maxi]
        best_epoch = results['epoch'][maxi]
    return best_epoch, best_test_acc

def parse_args():
    parser = argparse.ArgumentParser()
    # data I/O
    parser.add_argument('-data', '--dataset', type=str, default='Cifar-10',
                        help='Which dataset to use')
    parser.add_argument('-batch', '--batch_size', type=int, default=128,
                        help='Batch Size')
    parser.add_argument('-tbatch', '--test_batch_size', type=int, default=100,
                        help='Test Batch Size')
    parser.add_argument('-log', '--train_log_interval', type=int, default=50,
                        help='Log training interval every that often batches')
    parser.add_argument('-x', '--max_epochs', type=int, default=200,
                        help='How many args.max_epochs to run in total?')
    parser.add_argument('-s', '--seed', type=int, default=20181002,
                        help='Random seed to use')
    parser.add_argument('-aug', '--data_aug', type=str, nargs='+',
                        default=['random_h_flip', 'random_crop'],
                        help='Data augmentation?')
    # logging
    parser.add_argument('-en', '--exp_name', type=str, default='tester',
                        help='Experiment name for the model to be assessed')
    parser.add_argument('-o', '--logs_path', type=str, default='log',
                        help='Directory to save log files, check points, and tensorboard.')
    parser.add_argument('-resume', '--resume', type=int, default=0,
                        help='Resume training?')




    # THIS IMPLEMENTATION ##############################################################################################

    parser.add_argument('-np', '--num_paths', type=int, default=2,
                        help='Number of paths taken')
    parser.add_argument('-fc', '--use_fc', type=int, default=0,
                        help='Use FCs')
    parser.add_argument('-mult', '--sim_loss_mult', type=float, default=-1,
                        help='Base learning rate')
    parser.add_argument('-logitloss', '--logit_loss', type=int, default=0,
                        help='MSE loss in logits')
    parser.add_argument('-regmult', '--regularise_mult', type=float, default=0,
                        help='Regularise on input')

    parser.add_argument('-ups', '--upsample_type', type=str, default='pixel',
                        help='Upsample type for gen')
    parser.add_argument('-ca', '--classify_augmentations', type=int, default=0,
                        help='Classify augmentations too?')

    parser.add_argument('-cut', '--learn_cutout', type=int, default=0,
                        help='Learn the correct cutout?')

    parser.add_argument('-smt', '--softmax_type', type=str, default='gumbel',
                        help='Type of softmax')


    # model
    parser.add_argument('-model', '--model', type=str, default='preact_resnet',
                        help='resnet | preact_resnet | densenet | wresnet')
    ############### Resnet model stuff:
    parser.add_argument('-dep', '--resdepth', type=int, default=10,
                        help='ResNet default depth')
    parser.add_argument('-wf', '--widen_factor', type=int, default=2,
                        help='Wide resnet widen factor')


    ############### Simple model stuff:
    parser.add_argument('-act', '--activation', type=str, default='leaky_relu',
                        help='Activation function')
    parser.add_argument('-f', '--filters', type=int, nargs='+', default=[64, 128, 256, 512],
                        help='Filters')
    # optimization
    parser.add_argument('-l', '--learning_rate', type=float, default=0.1,
                        help='Base learning rate')
    parser.add_argument('-sched', '--scheduler', type=str, default='MultiStep',
                        help='Scheduler for learning rate annealing: CosineAnnealing | MultiStep')
    parser.add_argument('-mile', '--milestones', type=int, nargs='+', default=[60, 120, 160],
                        help='Multi step scheduler annealing milestones')

    args = parser.parse_args()
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))
    return args