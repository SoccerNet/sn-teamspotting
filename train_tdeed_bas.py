#!/usr/bin/env python3
"""
File containing the main training script to train T-DEED for SN-BAS challenge 2025.
"""

#Standard imports
import argparse
import os
import time
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
import wandb
import sys


#Local imports
from util.io import load_json, store_json, load_text
from dataset.datasets import get_datasets
from model.model import TDEEDModel
from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR)
from util.eval import mAPevaluate, mAPevaluateTest
from dataset.frame import ActionSpotVideoDataset


#Constants
EVAL_SPLITS = ['test']
STRIDE = 1
STRIDE_SN = 12
STRIDE_SNB = 2

def get_args():
    #Basic arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('-ag', '--acc_grad_iter', type=int, default=1,
                        help='Use gradient accumulation')
    parser.add_argument('--seed', type=int, default=1)
    return parser.parse_args()

def update_args(args, config):
    #Update arguments with config file
    args.frame_dir = config['frame_dir']
    args.save_dir = config['save_dir'] + '/' + args.model # + '-' + str(args.seed) -> in case multiple seeds
    args.store_dir = os.path.join(config['save_dir'], 'StoreClips', config['dataset']) #where to store clips information
    args.store_mode = config['store_mode']
    args.batch_size = config['batch_size']
    args.clip_len = config['clip_len']
    args.crop_dim = config['crop_dim']
    args.dataset = config['dataset']
    args.event_team = config['event_team']
    args.radi_displacement = config['radi_displacement']
    args.epoch_num_frames = config['epoch_num_frames']
    args.feature_arch = config['feature_arch']
    args.learning_rate = config['learning_rate']
    args.mixup = config['mixup']
    args.modality = config['modality']
    args.num_classes = config['num_classes']
    args.num_epochs = config['num_epochs']
    args.warm_up_epochs = config['warm_up_epochs']
    args.start_val_epoch = config['start_val_epoch']
    args.temporal_arch = config['temporal_arch']
    args.n_layers = config['n_layers']
    args.sgp_ks = config['sgp_ks']
    args.sgp_r = config['sgp_r']
    args.only_test = config['only_test']
    args.criterion = config['criterion']
    args.num_workers = config['num_workers']
    if 'joint_train' in config:
        args.joint_train = config['joint_train']
        args.joint_train['store_dir'] = os.path.join(args.save_dir, 'StoreClips', args.joint_train['dataset'])
    else:
        args.joint_train = None
    return args

def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    return args.num_epochs, ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer,
            num_steps_per_epoch * cosine_epochs)])


def main(args):
    #Set seed
    print('Setting seed to: ', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config_path = args.model.split('_')[0] + '/' + args.model + '.json'
    config = load_json(os.path.join('config', config_path))
    args = update_args(args, config)

    #Variables for SN & SNB label paths if datastes
    if (args.dataset == 'soccernet') | (args.dataset == 'soccernetball'):
        global LABELS_SN_PATH
        global LABELS_SNB_PATH
        LABELS_SN_PATH = load_text(os.path.join('data', 'soccernet', 'labels_path.txt'))[0]
        LABELS_SNB_PATH = load_text(os.path.join('data', 'soccernetball', 'labels_path.txt'))[0]

    assert args.batch_size % args.acc_grad_iter == 0
    if args.crop_dim <= 0:
        args.crop_dim = None

    # initialize wandb
    wandb.login()
    if not os.path.exists(args.save_dir + '/wandb_logs'):
        os.makedirs(args.save_dir + '/wandb_logs', exist_ok=True)
    wandb.init(config = args, dir = args.save_dir + '/wandb_logs', project = 'TDEED-snbas2025', name = args.model + '-' + str(args.seed))

    print("getting dataset from args: " + str(args))
    
    # Get datasets train, validation (and validation for map -> Video dataset)
    classes, joint_train_classes, train_data, val_data, val_data_frames = get_datasets(args)

    if args.store_mode == 'store':
        print('Datasets have been stored correctly! Stop training here and rerun.')
        sys.exit('Datasets have correctly been stored! Stop training here and rerun with load mode.')
    else:
        print('Datasets have been loaded from previous versions correctly!')

    def worker_init_fn(id):
        random.seed(id + epoch * 100)
    loader_batch_size = args.batch_size // args.acc_grad_iter

    # Dataloaders
    # train_loader = DataLoader(
    #     train_data, shuffle=False, batch_size=loader_batch_size,
    #     pin_memory=True, num_workers=args.num_workers,
    #     prefetch_factor=2, worker_init_fn=worker_init_fn)
        
    # val_loader = DataLoader(
    #     val_data, shuffle=False, batch_size=loader_batch_size,
    #     pin_memory=True, num_workers=args.num_workers,
    #     prefetch_factor=2, worker_init_fn=worker_init_fn)
                
    # Model
    model = TDEEDModel(args=args)

    #If joint_train -> 2 prediction heads
    if args.joint_train != None:
        n_classes = [len(classes)//2+1, len(joint_train_classes)//2+1]
        model._model.update_pred_head(n_classes)
        model._num_classes = np.array(n_classes).sum() 

    # optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})

    # if not args.only_test:
    #     # Warmup schedule
    #     num_steps_per_epoch = len(train_loader) // args.acc_grad_iter
    #     num_epochs, lr_scheduler = get_lr_scheduler(
    #         args, optimizer, num_steps_per_epoch)
        
    #     losses = []
    #     best_criterion = 0 if args.criterion == 'map' else float('inf')
    #     epoch = 0

    #     print('START TRAINING EPOCHS')
    #     for epoch in range(epoch, num_epochs):

    #         time_train0 = time.time()
    #         train_losses = model.epoch(
    #             train_loader, optimizer, scaler,
    #             lr_scheduler=lr_scheduler, acc_grad_iter=args.acc_grad_iter)
    #         train_loss = train_losses['loss']
    #         time_train1 = time.time()
    #         time_train = time_train1 - time_train0
            
    #         time_val0 = time.time()
    #         val_losses = model.epoch(val_loader, acc_grad_iter=args.acc_grad_iter)
    #         val_loss = val_losses['loss']
    #         time_val1 = time.time()
    #         time_val = time_val1 - time_val0

    #         better = False
    #         val_mAP = 0
    #         if args.criterion == 'loss':
    #             if val_loss < best_criterion:
    #                 best_criterion = val_loss
    #                 better = True
    #         elif args.criterion == 'map':
    #             time_map = 0
    #             if epoch >= args.start_val_epoch:
    #                 time_map0 = time.time()
    #                 val_mAP = mAPevaluate(model, val_data_frames, classes, printed=True, event_team = args.event_team, metric = 'at1')
    #                 time_map1 = time.time()
    #                 time_map = time_map1 - time_map0
    #                 if val_mAP > best_criterion:
    #                     best_criterion = val_mAP
    #                     better = True
            
    #         #Printing info epoch
    #         print('[Epoch {}] Train loss: {:0.5f} Val loss: {:0.5f}'.format(
    #             epoch, train_loss, val_loss))
    #         txt_losses_train = 'Train losses - lossC: {:0.5f} '.format(train_losses['lossC'])
    #         txt_losses_val = 'Val losses - lossC: {:0.5f} '.format(val_losses['lossC'])
    #         if 'lossD' in train_losses.keys():
    #             txt_losses_train += '- lossD: {:0.5f} '.format(train_losses['lossD']) 
    #             txt_losses_val += '- lossD: {:0.5f} '.format(val_losses['lossD'])
    #         if 'lossT' in train_losses.keys():
    #             txt_losses_train += '- lossT: {:0.5f} '.format(train_losses['lossT'])
    #             txt_losses_val += '- lossT: {:0.5f} '.format(val_losses['lossT'])
    #         print(txt_losses_train)
    #         print(txt_losses_val)
    #         if (args.criterion == 'map') & (epoch >= args.start_val_epoch):
    #             print('Val mAP: {:0.5f}'.format(val_mAP))
    #             if better:
    #                 print('New best mAP epoch!')
    #         print('Time train: ' + str(int(time_train // 60)) + 'min ' + str(np.round(time_train % 60, 2)) + 'sec')
    #         print('Time val: ' + str(int(time_val // 60)) + 'min ' + str(np.round(time_val % 60, 2)) + 'sec')
    #         if (args.criterion == 'map') & (epoch >= args.start_val_epoch):
    #             print('Time map: ' + str(int(time_map // 60)) + 'min ' + str(np.round(time_map % 60, 2)) + 'sec')

    #         losses.append({
    #             'epoch': epoch, 'train': train_loss, 'val': val_loss,
    #             'val_mAP': val_mAP
    #         })

    #         # Log to wandb
    #         if (args.criterion == 'map'):
    #             wandb.log({'losses/train/loss': train_loss, 'losses/val/loss': val_loss, 'losses/val/mAP': val_mAP, 'times/time_train': time_train, 'times/time_val': time_val, 'times/time_map': time_map})
    #         else:
    #             wandb.log({'losses/train/loss': train_loss, 'losses/val/loss': val_loss, 'times/time_train': time_train, 'times/time_val': time_val})

    #         if (args.radi_displacement > 0) & (args.event_team):
    #             wandb.log({'losses/train/lossC': train_losses['lossC'], 'losses/train/lossD': train_losses['lossD'], 'losses/train/lossT': train_losses['lossT'], 'losses/val/lossC': val_losses['lossC'], 'losses/val/lossD': val_losses['lossD'], 'losses/val/lossT': val_losses['lossT']})
    #         elif (args.radi_displacement > 0):
    #             wandb.log({'losses/train/lossC': train_losses['lossC'], 'losses/train/lossD': train_losses['lossD'], 'losses/val/lossC': val_losses['lossC'], 'losses/val/lossD': val_losses['lossD']})
    #         elif (args.event_team):
    #             wandb.log({'losses/train/lossC': train_losses['lossC'], 'losses/train/lossT': train_losses['lossT'], 'losses/val/lossC': val_losses['lossC'], 'losses/val/lossT': val_losses['lossT']})
    #         else:
    #             wandb.log({'losses/train/lossC': train_losses['lossC'], 'losses/val/lossC': val_losses['lossC']})


    #         if args.save_dir is not None:
    #             os.makedirs(args.save_dir, exist_ok=True)
    #             store_json(os.path.join(args.save_dir, 'loss.json'), losses,
    #                         pretty=True)

    #             if better:
    #                 torch.save(
    #                     model.state_dict(),
    #                     os.path.join(args.save_dir, 'checkpoint_best.pt'))

    print('START INFERENCE')
    model.load(torch.load(os.path.join(
        args.save_dir, 'checkpoint_best.pt')))

    eval_splits = EVAL_SPLITS
    inv_classes = {v: k for k, v in classes.items()}

    for split in eval_splits:
        split_path = os.path.join(
            'data', args.dataset, '{}.json'.format(split))
        
        stride = STRIDE
        if args.dataset == 'soccernet':
            stride = STRIDE_SN
        if args.dataset == 'soccernetball':
            stride = STRIDE_SNB

        if not os.path.exists(split_path):
            print('Split {} does not exist'.format(split))
            continue

        print("classes: " + str(classes))
        print("split_path: " + str(split_path))
        print("frame_dir: " + str(args.frame_dir))
        
        split_data = ActionSpotVideoDataset(classes, split_path, args.frame_dir, args.modality,
                # args.clip_len, overlap_len = 0, stride = stride, dataset = args.dataset, event_team = args.event_team)
                args.clip_len, overlap_len = args.clip_len // 4 * 3, stride = stride, dataset = args.dataset, event_team = args.event_team)

        print("split_data: " + str(split_data))
        print("dataset: " + str(args.dataset))
        
        pred_file = None
        if args.save_dir is not None:
            pred_file = os.path.join(args.save_dir, 'pred-{}'.format(split))

        results = mAPevaluateTest(model, split, split_data, classes, printed = True, event_team = args.event_team,
                metric = 'at1', pred_file = pred_file, postprocessing = 'SNMS')
        
        if results == None:
            print('No results for split {}'.format(split))
            print('Predictions have been stored in {}'.format(pred_file))
            continue
        
        wandb.log({'test/mAP@1': results['mAP'] * 100})
        wandb.summary['test/mAP@1'] = results['mAP'] * 100

        for j in range(len(classes) // 2):
            wandb.log({'test/classes/mAP@' + inv_classes[j*2+1].split('-')[0]: results['mAP_per_class'][j] * 100})

        if args.event_team:
            wandb.log({'test/mAP@1NoTeam': results['mAP_no_team'] * 100})
            wandb.summary['test/mAP@1NoTeam'] = results['mAP_no_team'] * 100

            for j in range(len(classes) // 2):
                wandb.log({'test/classes/mAP@' + inv_classes[j*2+1].split('-')[0] + 'NoTeam': results['mAP_per_class_no_team'][j] * 100})

    print('CORRECTLY FINISHED TRAINING AND INFERENCE')




if __name__ == '__main__':
    main(get_args())