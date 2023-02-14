import torch

from pytoda.datasets import AnnotatedDataset, SMILESTokenizerDataset
from toxsmi.utils import disable_rdkit_logging

"""we have:
1) train_scores_filepath    -   Path to the training toxicity scores (.csv)
2) test_scores_filepath     -   Path to the test toxicity scores (.csv)
3) smi_filepath             -   Path to the SMILES data (.smi)
5) model_path               -   Directory where the model will be stored.
6) params_filepath          -   Path to the parameter file.
7) training_name            -   Name for the training.
--embedding_path            -   Optional path to a pickle object of a pretrained embedding."""


def build_lookup(params: dict, smi_filepath):
    disable_rdkit_logging()
    smiles_dataset = SMILESTokenizerDataset(smi_filepath,
        padding_length=params.get("smiles_padding_length", None),
        padding=params.get("padd_smiles", True),
        add_start_and_stop=params.get("add_start_stop_token", True),
        augment=params.get("augment_smiles", False),
        canonical=params.get("canonical", False),
        kekulize=params.get("kekulize", False),
        all_bonds_explicit=params.get("all_bonds_explicit", False),
        all_hs_explicit=params.get("all_hs_explicit", False),
        randomize=params.get("randomize", False),
        remove_bonddir=params.get("remove_bonddir", False),
        remove_chirality=params.get("remove_chirality", False),
        selfies=params.get("selfies", False),
        sanitize=params.get("sanitize", True)
    )
    return smiles_dataset

def build_dataset( annotations_filepath, smi_lookup_dataset) -> torch.utils.data.DataLoader:
    # include arg label_columns if data file has any unwanted columns (such as index) to be ignored.
    dataset = AnnotatedDataset(
        annotations_filepath=annotations_filepath,
        dataset=smi_lookup_dataset
    )
    return dataset