# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from collections import OrderedDict
from easydict import EasyDict
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from datasets.video_datasets import build_dataset
from datasets.kinetics import build_training_dataset

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.models import create_model
import util.misc as misc
from util.pos_embed import interpolate_pos_embed_ori as interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from engine_finetune import train_one_epoch, evaluate
from engine_finetune import merge, final_test
import models


def construct_optimizer(model, args):
    # Batchnorm parameters.
    bn_params = []
    # Non-batchnorm parameters.
    non_bn_parameters = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if "bn" in name:
                bn_params.append(p)
            else:
                non_bn_parameters.append(p)
    optim_params = [
        {"params": bn_params, "weight_decay": 0.},
        {"params": non_bn_parameters, "weight_decay": args.weight_decay},
    ]
    return torch.optim.SGD(
        optim_params,
        lr=args.lr, weight_decay=args.weight_decay, momentum=0.9,
    )


def get_args_parser():
    parser = argparse.ArgumentParser('AdaptFormer fine-tuning for action recognition', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=174, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # custom parameters
    parser.add_argument('--linprob', default=True)
    parser.add_argument('--tubelet_size', type=int, default=2)
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='No drop path for linear probe')
    parser.add_argument('--use_mean_pooling', default=True)
    parser.add_argument('--init_scale', default=0.001, type=float)

    # video data parameters
    parser.add_argument('--data_set', default='SSV2',
                        choices=['SSV2', 'HMDB51', 'image_folder'],
                        type=str, help='dataset')
    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--num_sample', type=int, default=1,
                        help='Repeated_aug (default: 1)')
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--test_num_segment', type=int, default=4)
    parser.add_argument('--test_num_crop', type=int, default=3)
    parser.add_argument('--input_size', default=224, type=int, help='videos input size')

    # AdaptFormer related parameters
    parser.add_argument('--ffn_adapt', default=False, action='store_true', help='whether activate AdaptFormer')
    parser.add_argument('--ffn_num', default=64, type=int, help='bottleneck middle dimension')
    parser.add_argument('--vpt', default=False, action='store_true', help='whether activate VPT')
    parser.add_argument('--vpt_num', default=1, type=int, help='number of VPT prompts')
    parser.add_argument('--fulltune', default=False, action='store_true', help='full finetune model')
    parser.add_argument('--inception', default=False, action='store_true', help='whether use INCPETION mean and std'
                                                                                '(for Jx provided IN-21K pretrain')

    return parser


def main(args):
    if args.log_dir is None:
        args.log_dir = args.output_dir
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # dataset_train, args.nb_classes = build_dataset(is_train=True, test_mode=False, args=args)
    if args.data_set == 'SSV2':
        args.nb_classes = 174
    elif args.data_set == 'HMDB51':
        args.nb_classes = 51
    else:
        raise ValueError(args.data_set)
    dataset_train = build_training_dataset(args)
    dataset_val, _ = build_dataset(is_train=False, test_mode=False, args=args)
    dataset_test, _ = build_dataset(is_train=False, test_mode=True, args=args)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # fine-tuning configs
    tuning_config = EasyDict(
        # AdaptFormer
        ffn_adapt=args.ffn_adapt,
        ffn_option="parallel",
        ffn_adapter_layernorm_option="none",
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="0.1",
        ffn_num=args.ffn_num,
        d_model=768,
        # VPT related
        vpt_on=args.vpt,
        vpt_num=args.vpt_num,
    )

    if args.model.startswith('swin_'):
        # Video Swin
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            tuning_config=tuning_config,
        )
    else:
        model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        all_frames=args.num_frames * args.num_segments,
        tubelet_size=args.tubelet_size,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        tuning_config=tuning_config,
        )
        patch_size = model.patch_embed.patch_size
        print("Patch size = %s" % str(patch_size))
        args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
        args.patch_size = patch_size

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        if 'model' in checkpoint:
            raw_checkpoint_model = checkpoint['model']
        elif 'module' in checkpoint:
            raw_checkpoint_model = checkpoint['module']
        else:
            raw_checkpoint_model = checkpoint

        # TODO: refine
        if os.path.basename(args.finetune).startswith('pretrain'):
            checkpoint_model = OrderedDict()
            for k, v in raw_checkpoint_model.items():
                if k.startswith('encoder.'):
                    checkpoint_model[k[8:]] = v  # remove 'encoder.' prefix
            del checkpoint_model['norm.weight']
            del checkpoint_model['norm.bias']
        elif os.path.basename(args.finetune).startswith('finetune'):
            checkpoint_model = raw_checkpoint_model
        elif os.path.basename(args.finetune) == "vit_base_patch16_224_in21k_tongzhan_new.pth":
            checkpoint_model = raw_checkpoint_model
            del checkpoint_model['norm.weight']
            del checkpoint_model['norm.bias']
        elif os.path.basename(args.finetune).startswith('swin_base_patch244'):
            checkpoint_model = OrderedDict()
            for k, v in raw_checkpoint_model['state_dict'].items():
                if k.startswith('backbone.'):
                    checkpoint_model[k[9:]] = v
        else:
            raise ValueError("Warning: Double Check!")

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)

    # hack: revise model's head with BN
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    if not args.resume:
        # freeze all but the head
        for name, p in model.named_parameters():
            if name in msg.missing_keys:
                p.requires_grad = True
            else:
                p.requires_grad = False if not args.fulltune else True
        for _, p in model.head.named_parameters():
            p.requires_grad = True
        for _, p in model.fc_norm.named_parameters():
            p.requires_grad = True

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    optimizer = construct_optimizer(model_without_ddp, args)
    print(optimizer)
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')
        test_stats = final_test(data_loader_test, model, device, preds_file)
        torch.distributed.barrier()
        if global_rank == 0:
            print("Start merging results...")
            final_top1, final_top5 = merge(args.output_dir, num_tasks, is_hmdb=args.data_set == 'HMDB51')
            print(f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
            log_stats = {'Final top-1': final_top1, 'Final Top-5': final_top1}
            if args.output_dir and misc.is_main_process():
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')
    test_stats = final_test(data_loader_test, model, device, preds_file)
    torch.distributed.barrier()
    if global_rank == 0:
        print("Start merging results...")
        final_top1, final_top5 = merge(args.output_dir, num_tasks, is_hmdb=args.data_set == 'HMDB51')
        print(f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
        log_stats = {'Final top-1': final_top1, 'Final Top-5': final_top5}
        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
