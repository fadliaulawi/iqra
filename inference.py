import os
# Disable torchcodec to avoid FFmpeg dependency - set BEFORE importing datasets
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"

import torch
import torchaudio
import s3prl
import warnings
import argparse
from pathlib import Path
import tempfile
import requests
from tqdm import tqdm

warnings.filterwarnings("ignore")

#@title S3PRLModel utility code (click to expand)
def download_if_needed(path):
    """Download file if path is a URL, else return path unchanged."""
    if str(path).startswith("http://") or str(path).startswith("https://"):
        response = requests.get(path)
        response.raise_for_status()
        suffix = os.path.splitext(str(path))[-1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(response.content)
        tmp.close()
        return tmp.name
    return path

class S3PRLModel:
    def __init__(self, ckpt, dict_path='dict.txt'):
        from s3prl.downstream.runner import Runner

        # Support local or remote paths for ckpt and dict
        ckpt = download_if_needed(ckpt)
        dict_path = download_if_needed(dict_path)
        torch.serialization.add_safe_globals([argparse.Namespace])
        model_dict = torch.load(ckpt, map_location='cpu', weights_only=False)
        self.args = model_dict['Args']
        self.config = model_dict['Config']

        # Config patch
        self.args.init_ckpt = ckpt
        self.args.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config['downstream_expert']['text']["vocab_file"] = dict_path
        self.config['runner']['upstream_finetune'] = False
        self.config['runner']['layer_drop'] = False
        self.config['runner']['downstream_pretrained'] = None

        runner = Runner(self.args, self.config)
        self.upstream = runner._get_upstream()
        self.featurizer = runner._get_featurizer()
        self.downstream = runner._get_downstream()
        # For temp file cleanup
        self._temp_ckpt = ckpt if ckpt.startswith(tempfile.gettempdir()) else None
        self._temp_dict = dict_path if dict_path.startswith(tempfile.gettempdir()) else None

    def __call__(self, wav_path):
        wav, sr = torchaudio.load(wav_path)
        wav = wav.mean(0).unsqueeze(0)  # Convert to mono
        wav = wav.to(self.args.device)

        # Prepare dummy inputs
        dummy_split = "inference"
        dummy_filenames = [Path(wav_path).stem]  # Use filename as ID
        dummy_records = {"loss": [], "hypothesis": [], "groundtruth": [], "filename": []}

        with torch.no_grad():
            features = self.upstream.model(wav)
            features = self.featurizer.model(wav, features)
            dummy_labels = [[] for _ in features]  # Empty labels
            self.downstream.model(dummy_split, features, dummy_labels, dummy_filenames, dummy_records)
            predictions = dummy_records["hypothesis"]

        return predictions

    def cleanup(self):
        # Clean up downloaded temporary files
        for f in [self._temp_ckpt, self._temp_dict]:
            if f and os.path.isfile(f):
                os.remove(f)

def process_directory(ckpt, dict_path, wav_dir, output_csv):
    model = S3PRLModel(ckpt, dict_path)
    wav_files = list(Path(wav_dir).glob("*.wav"))
    wav_files.sort()
    results = []

    for wav_file in wav_files:
        output = model(str(wav_file))[0]
        print(f"{wav_file.name}: {output}")
        results.append({"ID": wav_file.name.split('.')[0], "Prediction": output})

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    # Clean up temp files if any
    model.cleanup()

# Download sample dataset and save groundtruth: Reference_phoneme and Annotation_phoneme
from datasets import load_dataset, Audio
import os
from scipy.io import wavfile
import pandas as pd
import soundfile as sf

# Download dataset
dataset = load_dataset("IqraEval/Iqra_Train", split="train")

# Cast audio column to disable automatic decoding (avoids FFmpeg dependency)
dataset = dataset.cast_column("audio", Audio(decode=False))

# Directory for wav files
wav_dir = "train_wavs"
os.makedirs(wav_dir, exist_ok=True)

# Prepare train.csv and save wav files
train_rows = []
for i, sample in tqdm(enumerate(dataset)):

    if i > 1000:
        break

    uid = sample['id']
    ref_phn = sample['phoneme_ref']
    # ann_phn = sample['Annotation_phoneme']
    train_rows.append({
        "ID": uid,
        "Reference_phn": ref_phn,
        # "Annotation_phn": ann_phn
    })
    wav_path = os.path.join(wav_dir, f"{uid}.wav")
    
    # Since decode=False, use bytes field and save directly
    audio_bytes = sample['audio']['bytes']
    # Write bytes to temporary file, then load with torchaudio
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name
    
    try:
        audio_data, sr = torchaudio.load(tmp_path)
        # Convert to mono and save
        audio_data = audio_data.mean(0, keepdim=True)
        torchaudio.save(wav_path, audio_data, sr)
    finally:
        os.remove(tmp_path)
    
print(f"Saved {len(train_rows)} wav files to {wav_dir}")

# Save train.csv in Colab root for the evaluation scripts
train_df = pd.DataFrame(train_rows)
train_df.to_csv("train.csv", index=False)
print("Prepared train.csv")

# Provide local path or URL to checkpoint
ckpt = "https://huggingface.co/IqraEval/Iqra_hubert_base/resolve/main/hubert_base.ckpt" # specify model, for e.g., wavlm_base.ckpt
dict_path = "interspeech_IqraEval/vocab/sws_arabic.txt"   # specify vocab
wav_dir = "train_wavs"   # e.g., "/content/wavs"
output_csv = "results.csv" # e.g., "/content/results.csv"

process_directory(ckpt, dict_path, wav_dir, output_csv)