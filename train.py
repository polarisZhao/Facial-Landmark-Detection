# -*- coding:utf-8 -*-
import argparse
import pprint
import time
import os

import torch
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from src import models
from src.datasets import FLDDatasets
from src.loss import WeightedLoss
from src.transforms import decode_preds, compute_nme
from src.utils import *

import logging
logger = logging.getLogger(__name__)


def train(config, train_loader, model, critertion, optimizer, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    nme_count = 0
    nme_batch_sum = 0

    end = time.time()
    for i, (inp, target, meta) in enumerate(train_loader):

        data_time.update(time.time() - end)
        output = model(inp)
        target = target.cuda(non_blocking=True)
        loss = critertion(output, target)

        score_map = output.data.cpu()
        preds = decode_preds(score_map, meta['center'], meta['scale'],
                             [64, 64])

        nme_batch = compute_nme(preds, meta)
        nme_batch_sum = nme_batch_sum + np.sum(nme_batch)
        nme_count = nme_count + preds.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inp.size(0))

        batch_time.update(time.time() - end)
        if i % config["PRINT_FREQ"] == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=inp.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

        end = time.time()
    nme = nme_batch_sum / nme_count
    msg = 'Train Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f}'\
        .format(epoch, batch_time.avg, losses.avg, nme)
    logger.info(msg)
    return losses.avg


def validate(config, val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_classes = config["MODEL"]["NUM_JOINTS"]
    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(val_loader):
            data_time.update(time.time() - end)
            output = model(inp)
            target = target.cuda(non_blocking=True)
            score_map = output.data.cpu()
            loss = criterion(output, target)
            preds = decode_preds(score_map, meta['center'], meta['scale'],
                                 [64, 64])
            # NME
            nme_temp = compute_nme(preds, meta)
            # Failure Rate under different threshold
            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)

            losses.update(loss.item(), inp.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(epoch, batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)
    return nme, losses.avg


def main(config, args):
    # env setting
    logger, final_output_dir, tb_log_dir = \
        create_logger(config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config["CUDNN"]["BENCHMARK"]
    cudnn.determinstic = config["CUDNN"]["DETERMINISTIC"]
    cudnn.enabled = config["CUDNN"]["ENABLED"]

    writer = SummaryWriter(log_dir=tb_log_dir)
    gpus = list(config["GPUS"])

    # model, criterion, optimizer, scheduler
    model = models.shufflenetModel()
    model = nn.DataParallel(model, device_ids=gpus).cuda()
    criterion = WeightedLoss().cuda()
    optimizer = get_optimizer(config, model)

    best_nme = 100
    last_epoch = config["TRAIN"]["BEGIN_EPOCH"]
    resume_epoch = config["TRAIN"]["RESUME_EPOCH"]
    if config["TRAIN"]["RESUME"]:
        model_state_file = os.path.join(
            final_output_dir, 'checkpoint_{}.pth'.format(resume_epoch))
        print("ssss")
        if os.path.isfile(model_state_file):
            checkpoint, last_epoch, best_nme = load_checkpoint(
                model_state_file, model, optimizer)
        else:
            print("=> no checkpoint found")

    lr_scheduler = get_scheduler(config, optimizer, last_epoch)

    # dataset & dataloader
    train_dataset = FLDDatasets(config, is_train=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config["TRAIN"]["BATCH_SIZE_PER_GPU"] * len(gpus),
        shuffle=config["TRAIN"]["SHUFFLE"],
        num_workers=config["WORKERS"],
        pin_memory=config["PIN_MEMORY"])

    val_dataset = FLDDatasets(config, is_train=False)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=config["TEST"]["BATCH_SIZE_PER_GPU"] * len(gpus),
        shuffle=False,
        num_workers=config["WORKERS"],
        pin_memory=config["PIN_MEMORY"])
    print("ddd")
    for epoch in range(last_epoch, config["TRAIN"]["END_EPOCH"]):

        train_loss = train(config, train_loader, model, criterion, optimizer,
                           epoch)

        nme, val_loss = validate(config, val_loader, model, criterion, epoch)
        lr_scheduler.step(nme)

        writer.add_scalars('data/loss', {
            'val loss': train_loss,
            'train loss': val_loss
        }, epoch)
        writer.add_scalar('data/nme', nme, epoch)

        is_best = nme < best_nme
        best_nme = min(nme, best_nme)

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        print("best:", is_best)

        state = {
            "state_dict": model,
            "epoch": epoch + 1,
            "best_nme": best_nme,
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(state, is_best, final_output_dir,
                        'checkpoint_{}.pth'.format(epoch))

    final_model_state_file = os.path.join(final_output_dir, 'final_state.pth')
    logger.info(
        'saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Face Landmrk Detection')
    parser.add_argument('--cfg',
                        help='experiment configuration filename',
                        required=True,
                        type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = configparse(args.cfg)
    main(config, args)
