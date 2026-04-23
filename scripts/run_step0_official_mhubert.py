"""Run official Iqra mHuBERT artifact (.ckpt) on local test wavs.

Outputs CSV with columns: ID, Prediction
"""

import argparse
import tempfile
from pathlib import Path

import pandas as pd
import requests
import torch
import torchaudio
import yaml
from tqdm import tqdm


def _download_if_url(path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        resp = requests.get(path_or_url, timeout=120)
        resp.raise_for_status()
        suffix = Path(path_or_url).suffix or ".bin"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(resp.content)
        tmp.close()
        return tmp.name
    return path_or_url


class S3PRLModel:
    def __init__(self, ckpt: str, dict_path: str):
        from s3prl.downstream.runner import Runner

        ckpt_local = _download_if_url(ckpt)
        dict_local = _download_if_url(dict_path)

        torch.serialization.add_safe_globals([argparse.Namespace])
        model_dict = torch.load(ckpt_local, map_location="cpu", weights_only=False)
        self.args = model_dict["Args"]
        self.config = model_dict["Config"]

        self.args.init_ckpt = ckpt_local
        self.args.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config["downstream_expert"]["text"]["vocab_file"] = dict_local
        self.config["runner"]["upstream_finetune"] = False
        self.config["runner"]["layer_drop"] = False
        self.config["runner"]["downstream_pretrained"] = None

        runner = Runner(self.args, self.config)
        self.upstream = runner._get_upstream()
        self.featurizer = runner._get_featurizer()
        self.downstream = runner._get_downstream()

    @torch.no_grad()
    def predict_wav(self, wav_path: str) -> str:
        wav, _ = torchaudio.load(wav_path)
        wav = wav.mean(0).unsqueeze(0).to(self.args.device)

        dummy_split = "inference"
        dummy_filenames = [Path(wav_path).stem]
        dummy_records = {"loss": [], "hypothesis": [], "groundtruth": [], "filename": []}

        features = self.upstream.model(wav)
        features = self.featurizer.model(wav, features)
        dummy_labels = [[] for _ in features]
        self.downstream.model(dummy_split, features, dummy_labels, dummy_filenames, dummy_records)
        return dummy_records["hypothesis"][0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/step0_official_mhubert.yaml")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = S3PRLModel(ckpt=cfg["ckpt"], dict_path=cfg["dict_path"])

    in_df = pd.read_csv(cfg["input_csv"])
    rows = []
    for _, row in tqdm(in_df.iterrows(), total=len(in_df), desc="Step0 official predict"):
        uid = str(row["ID"])
        wav_path = str(row["wav_path"])
        pred = model.predict_wav(wav_path)
        rows.append({"ID": uid, "Prediction": pred})

    out_path = Path(cfg["output_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved predictions: {out_path}")


if __name__ == "__main__":
    main()
