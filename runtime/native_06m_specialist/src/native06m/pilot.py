"""Explicit, non-deployable five-class pilot with one reviewed training geography."""

import argparse
import json
from pathlib import Path

from .train import run_training


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--config",type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(run_training(args.manifest,args.config,development_pilot=True),indent=2))
