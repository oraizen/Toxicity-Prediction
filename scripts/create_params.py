import json

def main(params_filepath, embedding_path=None) -> dict:

    
    params = {}
    with open(params_filepath) as fp:
        params.update(json.load(fp))

    if embedding_path:
        params['embedding_path'] = embedding_path

    return params