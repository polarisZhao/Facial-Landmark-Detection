# -*- coding:utf-8 -*-
import argparse
import cv2
import numpy as np

import torch
from torch import nn
import torchvision

from src import models
from src.mtcnn.detector import detect_faces
from src.transforms import decode_preds
from src.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(config, args):

    model = models.shufflenetModel()
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])
    gpus = list(config["GPUS"])
    model = nn.DataParallel(model, device_ids=gpus)

    # load model
    state_dict = torch.load(args.model_file, map_location=torch.device('cuda'))
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    else:
        model.module.load_state_dict(state_dict)
    model.eval()

    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret:
            print("Open Camera Failed !")
            break
        height, width = img.shape[:2]
        bounding_boxes, _ = detect_faces(img)
        for box in bounding_boxes:

            x1, y1, x2, y2 = (box[:4] + 0.5).astype(np.int32)
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            size = int(max([w, h]) * 1.1)
            cx = x1 + w // 2
            cy = y1 + h // 2
            x1 = cx - size // 2
            x2 = x1 + size
            y1 = cy - size // 2
            y2 = y1 + size

            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)

            cropped = img[y1:y2, x1:x2]

            if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx,
                                             cv2.BORDER_CONSTANT, 0)

            input = cv2.resize(cropped, (256, 256))
            input = transform(input).unsqueeze(0).to(device)
            output = model(input)

            score_map = output.data.cpu()
            center = torch.Tensor([[(x2 - x1) / 2, (y2 - y1) / 2]])
            sacle = torch.Tensor([max(w, h) / 200])
            preds = decode_preds(score_map, center, sacle, [64, 64])

            pre_landmark = preds[0]
            pre_landmark = pre_landmark.cpu().detach().numpy().reshape(
                -1, 2) - [dx, dy]
            for (x, y) in pre_landmark.astype(np.int32):
                cv2.circle(img, (x1 + x, y1 + y), 2, (0, 0, 255), -1)

        cv2.imshow('landmark detection result', img)
        if cv2.waitKey(10) == 27:
            break


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