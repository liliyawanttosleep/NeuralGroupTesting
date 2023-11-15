from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
# import resnet_design3 as models  # Design 3
# import resnet_design2 as models # Design 2
import argparse
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models
import numpy as np
import logging
import pickle
import pathlib
import random

'''model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
'''

# Tang: You should modify the following paths accordingly
dataset_path = 'F:/Summer_Project_Technion/Tasks/NeuralGroupTesting/data/dummy_group_testing_dataset'


best_acc1 = 0


def main():

    pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)

    ##################################
    # True group testing with fixed schedule
    ##################################

    val_dataset_list = []
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    for folder_idx in range(2):
        valdir = os.path.join(dataset_path, str(folder_idx), 'val')
        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        # The following code is made by Tang to show the modified images
        # Check labels of the validation dataset
        print("========================================")
        print("Tang's Modified Section: Printing Transformed Tensor Information")
        print("========================================")

        # Load a single image from val_dataset (Let's say the first image)
        img_index = 0
        img_tensor, label = val_dataset[img_index]

        # Print the label
        print(f"Label for the image at index {img_index}: {label}")

        # Print the shape of the tensor
        # The shape tells you the dimensions of the tensor. For an image, it's generally [C, H, W].
        print(f"Shape of the tensor: {img_tensor.shape}")

        # Print the data type of the tensor
        # It's good to know the data type for debugging and understanding the kind of data you're dealing with.
        print(f"Data type of the tensor: {img_tensor.dtype}")

        # Print the first few values of the tensor for a quick look
        # This gives you a snapshot of the kind of values your tensor contains.
        print(f"First few values of the tensor: {img_tensor[:, :2, :2]}")

        # Print the min and max values of the tensor
        # Knowing the range of values can help you understand if the tensor has been normalized or scaled.
        print(f"Min value in the tensor: {torch.min(img_tensor)}")
        print(f"Max value in the tensor: {torch.max(img_tensor)}")

        # Print the whole tensor
        # Be cautious with this if your tensor is large, as it could flood your output.
        print(f"Whole tensor: {img_tensor}")

        # Explanation
        # This tensor represents a single image from your validation set after it has been resized, center-cropped, and normalized.
        # Each value in this tensor is a pixel value for a particular channel (R, G, or B).

        print("========================================")
        print("End of Tang's Modified Section")
        print("========================================")

        print(val_dataset.imgs[:150])  # Print first 10 image paths and labels

        import matplotlib.pyplot as plt
        from PIL import Image

        # Function to display image
        def display_image(image, title):
            plt.imshow(image)
            plt.title(title)
            plt.axis('off')
            plt.show()

        # Select an example image from the validation dataset (Let's say the first image)
        img_index = 0
        img, label = val_dataset[img_index]
        filename = val_dataset.imgs[img_index][0].split(
            '\\')[-1]  # Extract filename from the full path

        # Convert the tensor back to a PIL Image for the modified image
        modified_img = transforms.ToPILImage()(img)

        # Load the original image directly using PIL for comparison
        original_img = Image.open(
            val_dataset.imgs[img_index][0]).convert("RGB")

        # Display the original and modified images
        display_image(original_img, f"Original - {filename} - Label: {label}")
        display_image(modified_img, f"Modified - {filename} - Label: {label}")

        # End of Tang's code
        print("No. {}, val_dataset {}".format(folder_idx, valdir),
              "dataset len:", len(val_dataset.samples))
        val_dataset_list.append(val_dataset)
        del val_dataset

    group_test_val_dataset = GroupTestDataset_val(
        val_dataset_list, None, split='val', valK=1)
    group_test_val_loader = torch.utils.data.DataLoader(
        group_test_val_dataset,
        batch_size=15, shuffle=False,
        num_workers=2, pin_memory=True,
        drop_last=False)
    criterion = nn.CrossEntropyLoss()
    validate(group_test_val_loader, None,
             criterion, None, dumpResult=True)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    ##################################
    # Logging setting
    ##################################

    logging.basicConfig(
        filename=os.path.join(args.output_dir, args.log_name),
        filemode='w',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    warnings.filterwarnings("ignore")

    print(str(args))
    logging.info(str(args))

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset_list = []
    val_dataset_list = []

    for folder_idx in range(args.task_num):
        traindir = os.path.join(args.data, str(folder_idx), 'train')
        valdir = os.path.join(args.data, str(folder_idx), 'val')
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        print("No. {}, traindir {}".format(folder_idx, traindir),
              "dataset len:", len(train_dataset.samples))

        train_dataset_list.append(train_dataset)
        del train_dataset

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        print("No. {}, val_dataset {}".format(folder_idx, valdir),
              "dataset len:", len(val_dataset.samples))
        val_dataset_list.append(val_dataset)
        del val_dataset

    ##################################
    # True group testing with fixed schedule
    ##################################
    group_test_val_dataset = GroupTestDataset_val(
        val_dataset_list, args, split='val')
    group_test_val_loader = torch.utils.data.DataLoader(
        group_test_val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.val_workers, pin_memory=True,
        drop_last=False)
    criterion = nn.CrossEntropyLoss()
    validate(group_test_val_loader, None,
             criterion, args, dumpResult=True)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        back_bone_model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        back_bone_model = models.__dict__[args.arch]()

    ##################################
    # Modificaiton Happens in the backbone model
    ##################################
    model = back_bone_model

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(
                (args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
            # raise NotImplementedError
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            raise NotImplementedError
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        raise NotImplementedError
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            ##################################
            # Current Going into This Branch
            ##################################
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer

    ##################################
    # Add Weights
    ##################################
    # weights = torch.tensor([1.0, 2.0], dtype=torch.float32)
    # weights = weights / weights.sum()
    # criterion = nn.CrossEntropyLoss(weight=weights).cuda(args.gpu)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            # args.start_epoch = checkpoint['epoch']
            # best_acc1 = checkpoint['best_acc1']
            # if args.gpu is not None:
            #     # best_acc1 may be from a checkpoint from a different GPU
            #     best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.evaluate:
        acc1 = validate(group_test_val_loader, model,
                        criterion, args, dumpResult=True)
        return

    ##################################
    # No need to shuffle val data
    ##################################
    # val_dataset = TaskCoalitionDataset_SuperImposing(val_dataset_list, args, split='val')
    # print( "len(val_dataset)", len(val_dataset) )
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.val_workers, pin_memory=True,
    #     drop_last=False)

    ##################################
    # Also construct a val single dataset
    ##################################
    # val_dataset_K1 = TaskCoalitionDataset_SuperImposing(val_dataset_list, args, split='val', valK=0)
    # val_loader_K1 = torch.utils.data.DataLoader(
    #     val_dataset_K1,
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.val_workers, pin_memory=True,
    #     drop_last=False)

    # if args.evaluate:
    #     # TODO: move it forward, and hack a dataset there.
    #     acc1 = validate(val_loader, model, criterion, args, dumpResult=True)
    #     validate(val_loader_K1, model, criterion, args, dumpResult=False)
    #     return

    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        # train(train_loader, model, criterion, optimizer, epoch, args)
        train(train_dataset_list, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        # acc1 = validate(val_loader, model, criterion, args, dumpResult=True)
        ##################################
        # Also validate single image
        ##################################
        # validate(val_loader_K1, model, criterion, args, dumpResult=False)

        ##################################
        # Test with fixed group testing schedule
        ##################################
        acc1 = validate(group_test_val_loader, model,
                        criterion, args, dumpResult=True)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args=args)


def train(train_dataset_list, model, criterion, optimizer, epoch, args):

    train_dataset = TaskCoalitionDataset_SuperImposing(
        train_dataset_list, args, split='train')
    print("len(train_dataset)", len(train_dataset))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        if args.distributed:  # added to support distributed
            train_sampler.set_epoch(epoch)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(
            train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,
        drop_last=True)

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.2e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        # [batch_time, losses, center_losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        ##################################
        # multi-instance learning
        ##################################
        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1 = accuracy(output, target, topk=(1, ))[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i == len(train_loader)-1:
            progress.display(i)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def validate(val_loader, model, criterion, args, dumpResult=False):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    # model.eval()

    with torch.no_grad():
        end = time.time()

        ##################################
        # Fields to be stored for postprocessing
        ##################################
        target_all = []
        pred_score_all = []

        for i, (images, target) in enumerate(val_loader):
            print("TANG: {}th target is {}\n".format(i, target))

            # compute output
            # output = model(images)
            output = torch.randn(target.shape[0], 2)
            num_to_correct = int(0.9 * target.shape[0])
            random_indices = random.sample(
                range(target.shape[0]), num_to_correct)
            print(f"Data type of output: {output.dtype}")
            # Tang: We are seeing 15 items here because batch_size is 15
            print(f"Data type of target: {target.dtype}")
            for idx in random_indices:
                correct_class = target[idx].item()
                # Since it's binary classification, the incorrect class is 1 - correct_class
                incorrect_class = 1 - correct_class
                # Make the correct class more likely
                output[idx, correct_class] += 10
                # Make the incorrect class less likely
                output[idx, incorrect_class] -= 10
            loss = criterion(output, target)

            # measure accuracy and record loss
            # acc1, acc5 = accuracy(output, target, topk=(1,5))
            acc1 = accuracy(output, target, topk=(1, ))[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     progress.display(i)

            ##################################
            # For analysis
            ##################################
            output_scores = torch.nn.functional.softmax(output, dim=-1)
            positive_scores = output_scores[:, 1]

            target_all.append(target.cpu().numpy())
            pred_score_all.append(positive_scores.cpu().numpy())

        target_all = np.concatenate(target_all, axis=0)
        pred_score_all = np.concatenate(pred_score_all, axis=0)

        if dumpResult is True:
            # with open(os.path.join( args.output_dir, 'model_validate_dump.pkl'), "wb") as pkl_file:
            with open(os.path.join('F:/Summer_Project_Technion/Tasks/NeuralGroupTesting/data/output_dir', 'model_validate_dump.pkl'), "wb") as pkl_file:
                pickle.dump({
                    "target_all": target_all,
                    "pred_score_all": pred_score_all,
                },
                    pkl_file
                )

        # a large analysis here
        pred_label = (pred_score_all > 0.5)
        print("accuracy {:.3f}".format(accuracy_score(target_all, pred_label)))
        # unique_classes = np.unique(y_true)
        # if len(unique_classes) > 1:
        # Calculate ROC AUC score
        #    roc_auc = roc_auc_score(y_true, y_score)
        # else:
        #   print(f"Only one class {unique_classes[0]} present in y_true. ROC AUC score cannot be calculated.")

        print("roc_auc_score {:.3f}".format(
            roc_auc_score(target_all, pred_score_all)))
        print("confusion_matrix\n{}".format(
            confusion_matrix(target_all, pred_label)))
        print("classification_report\n{}".format(
            classification_report(target_all, pred_label)))

        # TODO: this should also be done with the ProgressMeter
        # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #       .format(top1=top1, top5=top5))
        print('VAL * Acc@1 {top1.avg:.3f}'
              .format(top1=top1))

        # if is_main_process():
        logging.info("accuracy {:.3f}".format(
            accuracy_score(target_all, pred_label)))
        logging.info("roc_auc_score {:.3f}".format(
            roc_auc_score(target_all, pred_score_all)))
        logging.info("confusion_matrix\n{}".format(
            confusion_matrix(target_all, pred_label)))
        logging.info("classification_report\n{}".format(
            classification_report(target_all, pred_label)))
        logging.info('VAL * Acc@1 {top1.avg:.3f}'
                     .format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', args=None):

    torch.save(state, os.path.join(args.output_dir, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(args.output_dir, filename),
            os.path.join(args.output_dir, 'model_best.pth.tar')
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        ##################################
        # Save to logging
        ##################################
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    # lr = args.lr * (0.1 ** (epoch // 5)) # doesn not work
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        # Tang
        print("TANG: output is {}\n".format(output))
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


##################################
# K - mixing
# Implementation Assumption:
# - background tasks data num is at least twice as the positive data num
##################################

class TaskCoalitionDataset_SuperImposing(torch.utils.data.Dataset):

    def __init__(self, dataset_list, args, split, valK=None):

        assert split in ['train', 'val']

        first_dataset = dataset_list[0]
        self.loader = first_dataset.loader

        self.transform = first_dataset.transform
        assert first_dataset.target_transform is None

        self.classes = list()
        self.class_to_idx = dict()
        self.args = args

        self.task_num = len(dataset_list)

        ##################################
        # Get all normal data first
        ##################################
        positive_data_list = dataset_list[0].samples  # weapon
        normal_data_list = []  # normal classes in imagenet
        for _, ds in enumerate(dataset_list[1:]):
            samples_this_ds = ds.samples
            normal_data_list.extend(samples_this_ds)

        normal_data_list = np.random.permutation(normal_data_list)

        ##################################
        # Split for binary classification
        ##################################
        negative_data_list = normal_data_list[:len(positive_data_list)]

        ##################################
        # Redo Label Assignment and do mixing
        ##################################
        positive_target = 1
        positive_data_list = [[s[0], positive_target]
                              for s in positive_data_list]
        negative_target = 0
        negative_data_list = [[s[0], negative_target]
                              for s in negative_data_list]

        mixing_data_list = positive_data_list + negative_data_list
        # print("mixing_data_list[0]", mixing_data_list[0])
        if split != 'val':
            mixing_data_list = np.random.permutation(
                mixing_data_list)  # preserve order
        # print("mixing_data_list[0]", mixing_data_list[0])

        if split == 'train':
            # self.background_K = 1 #8-1 #(16-1) # 4
            self.background_K = self.args.background_K
        elif split == 'val':
            if valK is None:
                # self.background_K = 1 # 8-1 # (16-1) # 4 # 0: 96%, 1:96%, 4 - 94%, 9: 89%, 14:80.7% 19: 72%
                self.background_K = self.args.background_K
            else:
                self.background_K = valK

        # self.background_K = 0 # 4

        background_K_list = [None for _ in range(self.background_K)]
        for k_idx in range(self.background_K):
            k_idx_data_list = np.random.permutation(
                normal_data_list)[:len(mixing_data_list)]
            background_K_list[k_idx] = k_idx_data_list

        ##################################
        # Assign as members
        # A list of lists
        # - First list: data mixing - postive or negative
        # - Second list - (K+1) list: background K lists
        ##################################
        self.dataset_samples = [mixing_data_list] + background_K_list

        ##################################
        # Augment with normal training data
        # Already in the train loop
        ##################################
        # if split == 'train':
        #     for folder_idx in range(self.background_K + 1):
        #         # print("self.dataset_samples[folder_idx]", self.dataset_samples[folder_idx], type(self.dataset_samples[folder_idx]))
        #         self.dataset_samples[folder_idx] = np.concatenate( [self.dataset_samples[folder_idx], mixing_data_list], axis=0)
        #         # repeated. so that in same format

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        ##################################
        # load T images
        ##################################

        superimposed_images = 0
        mixing_folder_idx = 0
        # (path, target)
        target = int(self.dataset_samples[mixing_folder_idx][index][1])

        images_for_stack_list = []
        for folder_idx in range(self.background_K + 1):
            # for folder_idx in [0]:
            path, _ = self.dataset_samples[folder_idx][index]
            sample = self.loader(path)
            sample = self.transform(sample)
            images_for_stack_list.append(sample)

        # superimposed_images = superimposed_images / (self.background_K + 1)
        superimposed_images = torch.stack(images_for_stack_list)

        return superimposed_images, target

    def __len__(self):
        return len(self.dataset_samples[0])

##################################
# K - mixing
# Implementation Assumption:
# - background tasks data num is at least twice as the positive data num
##################################


class GroupTestDataset_val(torch.utils.data.Dataset):

    def __init__(self, dataset_list, args, split, valK=None):

        assert split in ['val']

        first_dataset = dataset_list[0]
        self.loader = first_dataset.loader

        self.transform = first_dataset.transform
        assert first_dataset.target_transform is None

        self.classes = list()
        self.class_to_idx = dict()
        self.args = args

        self.task_num = len(dataset_list)

        ##################################
        # Get all normal data first
        ##################################
        positive_data_list = dataset_list[0].samples  # weapon
        # print("Tang: positive_data_list_original is {}\n".format(positive_data_list))

        ##################################
        # Redo Label Assignment and do mixing
        # filter based on file name list
        ##################################
        import Constants
        positive_target = 1

        ##################################
        # Default Prevalence = 0.1%
        ##################################
        positive_data_list = [[s[0], positive_target] for s in positive_data_list if s[0].split(
            '\\')[-1] in Constants.firearm_file_paths]
        # assert len(positive_data_list) == 50, len(positive_data_list)

        normal_data_list = []  # normal classes in imagenet
        for _, ds in enumerate(dataset_list[1:]):
            samples_this_ds = ds.samples
            normal_data_list.extend(samples_this_ds)

        ##################################
        # Split for binary classification
        ##################################
        negative_data_list = normal_data_list

        ##################################
        # Overwrite to test on full test images (super noisy)
        ##################################
        # positive_data_list = [ [s[0], positive_target] for s in positive_data_list]
        # assert len(positive_data_list) == 150, len(positive_data_list)
        # negative_data_list = normal_data_list[:-100]

        ##################################
        # Adjust Prevalence
        # Default: prevalence_percentage = 0.1%
        ##################################

        # prevalence_percentage = 0.05 # 0.5 # 1.0 #

        prevalence_percentage = 0.1  # default

        DEFAULT_prevalence_percentage = 0.1
        if prevalence_percentage == DEFAULT_prevalence_percentage:
            # no modification is needed
            pass

        elif prevalence_percentage > DEFAULT_prevalence_percentage:

            positive_data_list = positive_data_list * \
                int(prevalence_percentage/DEFAULT_prevalence_percentage)
            num_negative_cutoff = len(positive_data_list) - 50
            negative_data_list = negative_data_list[num_negative_cutoff:]
            # OK, repeat positive. cut others
            assert len(positive_data_list) == 250  # 500 #
            pass
        elif prevalence_percentage == 0.01:
            # cut posivtive, repeat others

            assert prevalence_percentage == 0.01
            positive_data_list = positive_data_list[::10]  # half of the data
            negative_data_list = negative_data_list + \
                negative_data_list[:5]  # extend 25 data points
            assert len(positive_data_list) == 5

        elif prevalence_percentage == 0.05:
            # cut posivtive, repeat others

            assert prevalence_percentage == 0.05
            positive_data_list = positive_data_list[::2]  # half of the data
            negative_data_list = negative_data_list + \
                negative_data_list[:25]  # extend 25 data points

            assert len(positive_data_list) == 25

        ##################################
        # Redo Label Assignment and do mixing
        ##################################
        negative_target = 0
        negative_data_list = [[s[0], negative_target]
                              for s in negative_data_list]

        # concat and shuffle
        # print("Tang: positive_data_list is {}\n".format(positive_data_list))
        # print("Tang: negative_data_list is {}\n".format(negative_data_list))
        mixing_data_list = positive_data_list + negative_data_list
        print("Tang: mixing_data_list is {}\n".format(mixing_data_list))

        ##################################
        # Consistent: Use seed 42 for all experiments.
        # For non-adaptive testing that requires another seed: use 43 as the second seed.
        ##################################

        indices = torch.randperm(
            len(mixing_data_list), generator=torch.Generator().manual_seed(42)).tolist()
        print("Tang: Indices is {}\n".format(indices))
        # indices = torch.randperm( len(mixing_data_list), generator=torch.Generator().manual_seed(43)).tolist() # only for non-adaptive testing

        shuffled_mixing_data_list = np.array(mixing_data_list)[indices]
        print("Tang: shuffled_mixing_data_list is {}\n".format(
            shuffled_mixing_data_list))
        assert len(shuffled_mixing_data_list) == len(mixing_data_list)

        ##################################
        # Configure background K
        ##################################
        # if valK is None:
        # self.background_K = 1 # 8-1 # (16-1) # 4 # 0: 96%, 1:96%, 4 - 94%, 9: 89%, 14:80.7% 19: 72%
        # self.background_K = self.args.background_K
        self.background_K = valK

        ##################################
        # Reshape the list
        ##################################
        print("shuffled_mixing_data_list", shuffled_mixing_data_list.shape)
        self.dataset_samples = np.array(shuffled_mixing_data_list).reshape(
            len(shuffled_mixing_data_list)//(self.background_K + 1),
            self.background_K + 1,
            2
        ).transpose((1, 0, 2))

        print("Tang: dataset_samples is {}\n".format(self.dataset_samples))
        print("Tang：shuffled_mixing_data_list\n", self.dataset_samples.shape)

        # print("self.dataset_samples", self.dataset_samples.shape)
        # print("self.dataset_samples[:,x]", self.dataset_samples[:,37])
        # print("EXIT")
        # exit(0)
        # print("self.dataset_samples", self.dataset_samples)

        ##################################
        # Dump the group testing schedule
        ##################################
        with open(os.path.join('F:/Summer_Project_Technion/Tasks/NeuralGroupTesting/data/output_dir', 'val_schedule.npy'), "wb") as npy_file:
            np.save(npy_file, self.dataset_samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        ##################################
        # load T images
        ##################################

        superimposed_images = 0
        mixing_folder_idx = 0
        # (path, target)
        target = int(self.dataset_samples[mixing_folder_idx][index][1])

        images_for_stack_list = []
        for folder_idx in range(self.background_K + 1):
            # for folder_idx in [0]:
            path, target_this = self.dataset_samples[folder_idx][index]
            # Note: need to construct each target for group testing right now
            target = target or int(target_this)
            sample = self.loader(path)
            sample = self.transform(sample)
            images_for_stack_list.append(sample)

        # superimposed_images = superimposed_images / (self.background_K + 1)
        superimposed_images = torch.stack(images_for_stack_list)

        return superimposed_images, target

    def __len__(self):
        return len(self.dataset_samples[0])


if __name__ == '__main__':
    main()
