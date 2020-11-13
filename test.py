import os
import pprint
import argparse
import sys
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from src import models
from src.datasets import FLDDatasets
from src.transforms import decode_preds, compute_nme
from src.utils import *

import numpy as np

import logging
logger = logging.getLogger(__name__)


def inference(config, data_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_classes = config["MODEL"]["NUM_JOINTS"]
    predictions = torch.zeros((len(data_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(data_loader):
            data_time.update(time.time() - end)
            output = model(inp)
            score_map = output.data.cpu()
            preds = decode_preds(score_map, meta['center'], meta['scale'],
                                 [64, 64])

            # NME
            nme_temp = compute_nme(preds, meta)

            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Results time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    return nme, predictions


def main(config, args):
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config["CUDNN"]["BENCHMARK"]
    cudnn.determinstic = config["CUDNN"]["DETERMINISTIC"]
    cudnn.enabled = config["CUDNN"]["ENABLED"]

    model = models.shufflenetModel()

    gpus = list(config["GPUS"])
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # load model
    state_dict = torch.load(args.model_file)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    else:
        model.module.load_state_dict(state_dict)

    test_loader = DataLoader(dataset=FLDDatasets(config, is_train=False),
                             batch_size=config["TEST"]["BATCH_SIZE_PER_GPU"] *
                             len(gpus),
                             shuffle=False,
                             num_workers=config["WORKERS"],
                             pin_memory=config["PIN_MEMORY"])

    nme, predictions = inference(config, test_loader, model)

    torch.save(predictions, os.path.join(final_output_dir, 'predictions.pth'))


def parse_args():
    parser = argparse.ArgumentParser(description='Train Face Alignment')
    parser.add_argument('--cfg',
                        help='experiment configuration filename',
                        required=True,
                        type=str)
    parser.add_argument('--model-file',
                        help='model parameters',
                        required=True,
                        type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = configparse(args.cfg)
    main(config, args)
