import argparse
import json
import logging
import os, sys
from time import time

import numpy as np
from sklearn.metrics import (
    auc, average_precision_score, precision_recall_curve, roc_curve
)

from pytoda.datasets import AnnotatedDataset, SMILESDataset
from pytoda.smiles.smiles_language import SMILESLanguage
from pytoda.smiles.transforms import SMILESToMorganFingerprints
from pytoda.transforms import Compose, ToTensor
from toxsmi.models import MODEL_FACTORY
from toxsmi.utils import disable_rdkit_logging

import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    'train_scores_filepath',
    type=str,
    help='Path to the training toxicity scores (.csv)'
)
parser.add_argument(
    'test_scores_filepath',
    type=str,
    help='Path to the test toxicity scores (.csv)'
)
parser.add_argument(
    'smi_filepath', type=str, help='Path to the SMILES data (.smi)'
)
parser.add_argument(
    'smiles_language_filepath',
    type=str,
    help='Path to a pickle object a SMILES language object.'
)
parser.add_argument(
    'model_path', type=str, help='Directory where the model will be stored.'
)
parser.add_argument(
    'params_filepath', type=str, help='Path to the parameter file.'
)
parser.add_argument('training_name', type=str, help='Name for the training.')
parser.add_argument(
    '--embedding_path', type=str, default=None,
    help='Optional path to a pickle object of a pretrained embedding.'
)

def main(
    train_scores_filepath, test_scores_filepath, smi_filepath,
    smiles_language_filepath, model_path, params_filepath, training_name,
    embedding_path=None
):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(f'{training_name}')
    logger.setLevel(logging.INFO)