import json

def cfg_parser(cfg_file: str) -> dict:
    cfg = json.load(open(cfg_file))
    return cfg
