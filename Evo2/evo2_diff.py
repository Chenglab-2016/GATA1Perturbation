#!/usr/bin/env python3

import os
import json
import base64
from io import BytesIO

import numpy as np
import requests
from dotenv import load_dotenv


DNA_TO_INDEX = {
    "A": 65,
    "C": 67,
    "G": 71,
    "T": 84,
}


def decode_logits(base64_data):
    decoded = base64.b64decode(base64_data)
    npz = np.load(BytesIO(decoded))
    logits = npz["output_layer.output"]
    return np.squeeze(logits)


def calculate_sequence_score(seq, logits):

    total = 0.0

    for i in range(1, len(seq)):
        base = seq[i]

        if base not in DNA_TO_INDEX:
            continue

        idx = DNA_TO_INDEX[base]

        total += logits[i - 1][idx]

    return float(total)


class Evo2Client:

    def __init__(self, env_path="your_env_file_path"):  #### your env file includes PCAI_EVO2_ENDPOINT (local host Evo2 url) and PCAI_EVO2_TOKEN (user key)

        load_dotenv(env_path)

        self.endpoint = os.getenv("PCAI_EVO2_ENDPOINT")
        self.token = os.getenv("PCAI_EVO2_TOKEN")

        self.api = "/biology/arc/evo2/forward"

        self.headers = {
            "Authorization": f"Bearer {self.token}"
        }

    def score_sequence(self, seq):

        body = {
            "sequence": seq,
            "output_layers": ["output_layer"]
        }

        r = requests.post(
            f"{self.endpoint}{self.api}",
            headers=self.headers,
            json=body,
            verify=False
        )

        r.raise_for_status()

        logits = decode_logits(r.json()["data"])

        return calculate_sequence_score(seq, logits)


def compute_delta_scores(input_file, output_file):

    with open(input_file) as f:
        pairs = json.load(f)

    evo2 = Evo2Client()

    results = {}

    for w, (s1, s2) in pairs.items():

        score1 = evo2.score_sequence(s1)
        score2 = evo2.score_sequence(s2)

        delta = score2 - score1

        results[w] = delta

        print(w, delta)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)    ### set as ./data/json_for_evo2.json
    parser.add_argument("--output", default="delta_scores.json")

    args = parser.parse_args()

    compute_delta_scores(args.input, args.output)