# import package modules
from src.features import build_dataset, build_lookup
from src.models import setup_model,train_model

# import 3rd party modules
import torch

def main(params: dict, train_scores_filepath, test_scores_filepath, smi_filepath):
    smiles_lookup_dataset = build_lookup(params, smi_filepath)
    train_dataset = build_dataset( train_scores_filepath, smiles_lookup_dataset)
    test_dataset = build_dataset( test_scores_filepath, smiles_lookup_dataset)

    train_dataset_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        drop_last=False
    )
    test_dataset_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        drop_last=False
    )
    model = setup_model(params)
    train_model.train(model, train_dataset_loader, test_dataset_loader, smiles_lookup_dataset, params)