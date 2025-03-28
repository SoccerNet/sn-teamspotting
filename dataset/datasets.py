"""
File containing the function to load all the frame datasets.
"""

#Standard imports
import os

#Local imports
from util.dataset import load_classes
from dataset.frame import ActionSpotDataset, ActionSpotVideoDataset, ActionSpotDatasetJoint

#Constants
STRIDE = 1
STRIDE_SN = 12
STRIDE_SNB = 2
OVERLAP = 0.9
OVERLAP_SN = 0.50

def get_datasets(args):
    if args.event_team:
        classes = load_classes(os.path.join('data', args.dataset, 'class.txt'), event_team = True)
    else:
        classes = load_classes(os.path.join('data', args.dataset, 'class.txt'))

    print(classes)
    # return classes, None, None, None, None

    dataset_len = args.epoch_num_frames // args.clip_len
    stride = STRIDE
    overlap = OVERLAP
    if args.dataset == 'soccernet':
        stride = STRIDE_SN
        overlap = OVERLAP_SN
    elif args.dataset == 'soccernetball':
        stride = STRIDE_SNB
    

    dataset_kwargs = {
        'stride': stride, 'overlap': overlap, 'radi_displacement': args.radi_displacement,
        'mixup': args.mixup, 'dataset': args.dataset, 'event_team': args.event_team
    }

    print('Dataset size:', dataset_len)

    # train_data = ActionSpotDataset(
    #     classes, os.path.join('data', args.dataset, 'train.json'),
    #     args.frame_dir, args.store_dir, args.store_mode, 
    #     args.modality, args.clip_len, dataset_len, **dataset_kwargs)
    # train_data.print_info()
        
    dataset_kwargs['mixup'] = False # Disable mixup for validation

    # val_data = ActionSpotDataset(
    #     classes, os.path.join('data', args.dataset, 'val.json'),
    #     args.frame_dir, args.store_dir, args.store_mode,
    #     args.modality, args.clip_len, dataset_len // 4, **dataset_kwargs)
    # val_data.print_info()

    val_data_frames = None
    if args.criterion == 'map':
        # Only perform mAP evaluation during training if criterion is mAP
        val_data_frames = ActionSpotVideoDataset(
            classes, os.path.join('data', args.dataset, 'val.json'),
            args.frame_dir, args.modality, args.clip_len,
            overlap_len=0, stride = stride, dataset = args.dataset, event_team = args.event_team)        
        
    #In case of using joint_train, datasets with additional data
    joint_train_classes = None
    if args.joint_train != None:

        stride_joint_train = STRIDE
        overlap_joint_train = OVERLAP
        if args.joint_train['dataset'] == 'soccernet':
            stride_joint_train = STRIDE_SNB
            overlap_joint_train = OVERLAP_SN

        dataset_joint_train_kwargs = {
            'stride': stride_joint_train, 'overlap': overlap_joint_train, 'radi_displacement': args.radi_displacement,
            'mixup': args.mixup, 'dataset': args.joint_train['dataset'], 'event_team': args.event_team
        }

        if args.event_team:
            joint_train_classes = load_classes(os.path.join('data', args.joint_train['dataset'], 'class.txt'), event_team = True)
        else:
            joint_train_classes = load_classes(os.path.join('data', args.joint_train['dataset'], 'class.txt'))
        

        # joint_train_train_data = ActionSpotDataset(
        #     joint_train_classes, os.path.join('data', args.joint_train['dataset'], 'train.json'),
        #     args.joint_train['frame_dir'], args.joint_train['store_dir'], args.store_mode,
        #     args.modality, args.clip_len, dataset_len, **dataset_joint_train_kwargs)
        # joint_train_train_data.print_info()

        dataset_joint_train_kwargs['mixup'] = False # Disable mixup for validation

        # joint_train_val_data = ActionSpotDataset(
        #     joint_train_classes, os.path.join('data', args.joint_train['dataset'], 'val.json'),
        #     args.joint_train['frame_dir'], args.joint_train['store_dir'], args.store_mode,
        #     args.modality, args.clip_len, dataset_len // 4, **dataset_joint_train_kwargs)
        # joint_train_val_data.print_info()

        # train_data = ActionSpotDatasetJoint(train_data, joint_train_train_data)
        # val_data = ActionSpotDatasetJoint(val_data, joint_train_val_data)
    return classes, joint_train_classes, None, None, None        
    return classes, joint_train_classes, train_data, val_data, val_data_frames