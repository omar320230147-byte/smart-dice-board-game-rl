import argparse
from .experiments import run, evaluate_saved_models
from .plots import make_plots
from .play import main as play
p=argparse.ArgumentParser(prog="smart_dice_rl"); p.add_argument("command",choices=["train","evaluate","plots","play"]); args,rest=p.parse_known_args()
if args.command=="train": run()
elif args.command=="evaluate": evaluate_saved_models()
elif args.command=="plots": make_plots(__import__('pathlib').Path(__file__).resolve().parent.parent / "results")
else:
    import sys
    sys.argv=[sys.argv[0],*rest]; play()
