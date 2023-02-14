import sys
sys.path.append('C:/Development/Python/Thesis')

from src.features.build_features import main
from src.utils import get_device
import pandas as pd

params = {
    "stacked_dense_hidden_sizes": [
        1024,
        512
    ],
    "activation_fn": "relu",
    "dropout": 0.5,
    "batch_norm": True,
    "smiles_embedding_size": 256,
    "kernel_sizes": [
        [
            3,
            256
        ],
        [
            5,
            256
        ],
        [
            11,
            256
        ]
    ],
    "ensemble_size": 5,
    "ensemble": "score",
    "smiles_attention_size": 256,
    "embed_scale_grad": False,
    "batch_size": 256,
    "lr": 0.0001,
    "optimizer": "adam",
    "loss_fn": "binary_cross_entropy_ignore_nan_and_sum",
    "epochs": 200,
    "save_model": 25,
    "smiles_vocabulary_size": 87,
    "num_tasks": 1,
    "class_weights": [
        1,
        5
    ]
}
device = get_device()
loader = main(params, 'C:/Development/Python/Thesis/data/raw/tox21_score.csv', 'C:/Development/Python/Thesis/data/raw/tox21.smi')
print(len(loader.dataset))
for idx, batch in enumerate(loader):
    print(batch.smiles.size())
    print(f'outputs shape : {batch.y.shape}')
    break