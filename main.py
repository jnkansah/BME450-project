"""
Driver Drowsiness Detection — main entrypoint.

Usage:
  python main.py                   # webcam (device 0)
  python main.py --source 1        # webcam device 1
  python main.py --source video.mp4
  python main.py --train           # train all CNN models first
  python main.py --train --source 0
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Driver Drowsiness Detection System"
    )
    parser.add_argument("--source", default="0",
                        help="Webcam index (0) or path to video file")
    parser.add_argument("--train", action="store_true",
                        help="Train CNN models before running the detector")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs (used with --train)")
    parser.add_argument("--task", default="all",
                        choices=["eye", "mouth", "drowsy", "all"],
                        help="Which model to train (used with --train)")
    args = parser.parse_args()

    if args.train:
        from src.train import train_one_task
        tasks = ["eye", "mouth", "drowsy"] if args.task == "all" else [args.task]
        for task in tasks:
            train_one_task(task, args.epochs)

    from src.detector import run
    src = int(args.source) if args.source.isdigit() else args.source
    run(src)


if __name__ == "__main__":
    main()
