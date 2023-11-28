from typing import NoReturn
import sys
import argparse
import glob
import os
import torch

TRAIN = 'train'
EVALUATE = 'eval'

def ferror(msg: str) -> NoReturn:
    sys.stderr.write("Fatal: " + msg)
    quit(1)

parser = argparse.ArgumentParser(description="") # TODO: write more detailed description

parser.add_argument("-m", "--mode")
parser.add_argument("-d", "--device")
parser.add_argument("-r", "--directory")
parser.add_argument("-o", "--output")
parser.add_argument('-B', '--tensorboard', action='store_true')

args = parser.parse_args()
if args.mode == TRAIN:
    mode = TRAIN
elif args.mode == EVALUATE:
    mode = EVALUATE
else:
    ferror("Invalid mode %s. Accepted: 'train' to train, 'eval' to evaluate.\n" % str(args.mode))

device = args.device
in_drc = args.directory
out_drc = args.output
use_TB = args.tensorboard

if not os.path.exists(in_drc):
    ferror("%s is not a valid directory or filepath." % args.directory)
if not os.path.exists(out_drc):
    ferror("%s is not a valid directory or filepath." % args.directory)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    sys.stderr.write("MPS device not found. Using cuda.\n")
    device = torch.device("cuda")
else:
    sys.stderr.write("No GPU found. Using CPU. This may impact performance.\n")
    device = torch.device("cpu")








