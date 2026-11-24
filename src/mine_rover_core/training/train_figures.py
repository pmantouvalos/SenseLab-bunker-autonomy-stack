#!/usr/bin/env python3
"""
Trains the detector of the 5 figures (YOLOv8n) on the dataset made by collect_figure_dataset.py
and installs the result where the launch file looks for it.

    ~/venvs/yolo/bin/python src/mine_rover_core/training/train_figures.py \
        --data ~/datasets/figures/data.yaml --epochs 40

Result: src/mine_rover_core/models/figures.pt  (rebuild the workspace to install it).
Starts from the COCO-pretrained yolov8n.pt (transfer learning).

For the real toys: collect ~50 photos per toy with the robot camera, label them (e.g. with
Roboflow / labelImg), put them in the same folder structure, and run this again with --base
set to models/figures.pt (so it keeps what it learned in the simulator).
"""
import argparse
import os
import shutil

from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.expanduser('~/datasets/figures/data.yaml'))
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', default='0')
    ap.add_argument('--base', default='yolov8n.pt')
    ap.add_argument('--project', default=os.path.expanduser('~/runs/figures'))
    args = ap.parse_args()

    model = YOLO(args.base)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, project=args.project, name='figures', exist_ok=True,
                # characters can be seen from any side and at any size; the camera stays upright
                fliplr=0.5, degrees=5, translate=0.1, scale=0.5, mosaic=1.0,
                hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, workers=4, plots=True)
    best = os.path.join(args.project, 'figures', 'weights', 'best.pt')
    dest_dir = os.path.join(HERE, '..', 'models')
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.normpath(os.path.join(dest_dir, 'figures.pt'))
    shutil.copy(best, dest)
    print(f'\nInstalled {dest}')


if __name__ == '__main__':
    main()
