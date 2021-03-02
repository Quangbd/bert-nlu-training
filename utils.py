import os
import torch
import random
import logging
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score


def get_intent_labels(args):
    return [label.strip() for label in open(os.path.join(
        args.data_dir, 'cc', args.intent_label_file), 'r', encoding='utf-8')]


def get_slot_labels(args):
    return [label.strip() for label in open(os.path.join(
        args.data_dir, 'cc', args.slot_label_file), 'r', encoding='utf-8')]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
