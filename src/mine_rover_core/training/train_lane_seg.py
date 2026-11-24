#!/usr/bin/env python3
"""
Trains the lane-line segmentation network (YOLOv8n-seg) on the dataset made by
collect_lane_dataset.py and installs the result where lane_detector looks for it.

    ~/venvs/yolo/bin/python src/mine_rover_core/training/train_lane_seg.py \
        --data ~/datasets/lane_sim/data.yaml --epochs 40

Result: src/mine_rover_core/models/lane_seg.pt  (rebuild the workspace to install it).
Starts from the COCO-pretrained yolov8n-seg.pt (transfer learning), so a few hundred
images are enough for one class.
"""
import argparse
import os
import shutil

from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.expanduser('~/datasets/lane_sim/data.yaml'))
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', default='0')
    ap.add_argument('--base', default='yolov8n-seg.pt')
    ap.add_argument('--project', default=os.path.expanduser('~/runs/lane'))
    args = ap.parse_args()

    model = YOLO(args.base)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, project=args.project, name='lane_seg', exist_ok=True,
                # the camera never flips or rotates much: only mild geometric augmentation
                fliplr=0.5, degrees=5, translate=0.05, scale=0.2, mosaic=0.5,
                hsv_v=0.5, workers=4, plots=True)
    best = os.path.join(args.project, 'lane_seg', 'weights', 'best.pt')
    dest_dir = os.path.join(HERE, '..', 'models')
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.normpath(os.path.join(dest_dir, 'lane_seg.pt'))
    shutil.copy(best, dest)
    print(f'\nInstalled {dest}')


if __name__ == '__main__':
    main()
