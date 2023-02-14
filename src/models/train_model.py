import argparse
import json
import logging
import os, sys

import numpy as np
import torch
from sklearn.metrics import (
    auc, average_precision_score, precision_recall_curve, roc_curve
)

from src.utils import get_device, OPTIMIZER_FACTORY

from pytoda.smiles.transforms import SMILESToMorganFingerprints
from pytoda.transforms import Compose, ToTensor

"""we have:
1) train_scores_filepath    -   Path to the training toxicity scores (.csv)
2) test_scores_filepath     -   Path to the test toxicity scores (.csv)
3) smi_filepath             -   Path to the SMILES data (.smi)
4) smiles_language_filepath -   Path to a pickle object a SMILES language object.
5) model_path               -   Directory where the model will be stored.
6) params_filepath          -   Path to the parameter file.
7) training_name            -   Name for the training.
--embedding_path            -   Optional path to a pickle object of a pretrained embedding."""

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def train(model, train_dataset_loader, test_dataset_loader, smiles_dataset, params: dict):

    for name, param in model.named_parameters():
        logger.info((name, param.shape))

    if params.get("model_fn", "mca") == "dense":
        morgan_transform = Compose(
            [
                SMILESToMorganFingerprints(
                    radius=params.get("fp_radius", 2),
                    bits=params.get("num_drug_features", 512),
                    chirality=params.get("fp_chirality", True),
                ),
                ToTensor(),
            ]
        )

        def smiles_tensor_batch_to_fp(smiles):
            """To abuse SMILES dataset for FP usage"""
            out = torch.Tensor(smiles.shape[0], params.get("num_drug_features", 256))
            for ind, tensor in enumerate(smiles):
                smiles = smiles_dataset.smiles_language.token_indexes_to_smiles(tensor.tolist())
                out[ind, :] = torch.squeeze(morgan_transform(smiles))
            return out

    # Define optimizer
    optimizer = OPTIMIZER_FACTORY[params.get('optimizer', 'adam')](model.parameters(), lr=params.get('lr', 0.00001))
    
    logger.info('Training about to start...\n')
    min_loss, max_roc_auc = 1000000, 0
    max_precision_recall_score = 0
    device = get_device()
    for epoch in range(params['epochs']):
        model.train()
        train_loss = 0
        for ind, (smiles, y) in enumerate(train_dataset_loader):

            smiles = torch.squeeze(smiles)
            # Transform smiles to FP if needed
            if params.get('model_fn', 'mca') == 'dense':
                smiles = smiles_tensor_batch_to_fp(smiles).to(device)

            y_hat, pred_dict = model(smiles)

            loss = model.loss(y_hat, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        logger.info(
            '\t **** TRAINING ****   '
            f"Epoch [{epoch + 1}/{params['epochs']}], "
            f'loss: {train_loss / len(train_dataset_loader):.5f}. '
        )

        model.eval()
        with torch.no_grad():
            test_loss = 0
            predictions = []
            labels = []
            for ind, (smiles, y) in enumerate(test_dataset_loader):

                smiles = torch.squeeze(smiles.to(device))
                # Transform smiles to FP if needed
                if params.get('model_fn', 'mca') == 'dense':
                    smiles = smiles_tensor_batch_to_fp(smiles).to(device)

                y_hat, pred_dict = model(smiles)
                predictions.append(y_hat)
                # Copy y tensor since loss function applies downstream
                # modification
                labels.append(y.clone())
                loss = model.loss(y_hat, y.to(device)).cpu()
                test_loss += loss.item()
        
        predictions = torch.cat(predictions, dim=0).flatten().cpu().numpy()
        labels = torch.cat(labels, dim=0).flatten().cpu().numpy()

        # Remove NaNs from labels to compute scores
        predictions = predictions[~np.isnan(labels)]
        labels = labels[~np.isnan(labels)]
        test_loss_a = test_loss / len(test_dataset_loader)
        fpr, tpr, _ = roc_curve(labels, predictions)
        test_roc_auc_a = auc(fpr, tpr)

        # calculations for visualization plot
        precision, recall, _ = precision_recall_curve(labels, predictions)
        # score for precision vs accuracy
        test_precision_recall_score = average_precision_score(
            labels, predictions
        )