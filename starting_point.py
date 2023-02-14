import argparse, json, logging
from pathlib import Path

logging.basicConfig(level = logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("train_scores_filepath", type=str, help="Path to the training scores")
parser.add_argument("test_scores_filepath", type=str, help="Path to the test scores")
parser.add_argument("smi_filepath", type=str, help="Path to the smiles dataset file")
parser.add_argument("params_filepath", type=str, help="Path to the parameter file.")

def main(train_scores_filepath, test_scores_filepath, smi_filepath, params_filepath):
    import src.run
    with open(params_filepath) as p:
        params = {}
        params.update(json.load(p))
    # transform relative paths to absolute paths
    train_scores_filepath = Path(train_scores_filepath).resolve()
    test_scores_filepath = Path(test_scores_filepath).resolve()
    smi_filepath = Path(smi_filepath).resolve()
    src.run.main(params, train_scores_filepath, test_scores_filepath, smi_filepath)   

if __name__=="__main__":
    args = parser.parse_args()
    main(args.train_scores_filepath, args.test_scores_filepath, args.smi_filepath, args.params_filepath)