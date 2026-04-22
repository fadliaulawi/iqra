"""Whisper encoder-only + CTC head."""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, WhisperFeatureExtractor

from src.data.augmentation import apply_specaugment


class WhisperEncoderCTC(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        num_labels: int,
        blank_id: int,
        head_hidden: int = 1024,
        head_layers: int = 2,
        head_dropout: float = 0.1,
        gradient_checkpointing: bool = False,
        freeze_encoder: bool = False,
        specaugment: bool = False,
    ):
        super().__init__()
        self.blank_id = blank_id
        self.freeze_encoder = freeze_encoder
        self.use_specaugment = specaugment

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(encoder_name)
        self.model = AutoModel.from_pretrained(encoder_name, torch_dtype=torch.float32)  # WhisperModel
        self.encoder = self.model.encoder

        for p in self.encoder.parameters():
            p.requires_grad = not self.freeze_encoder

        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

        hidden_size = int(self.model.config.d_model)
        self.head = nn.LSTM(
            input_size=hidden_size,
            hidden_size=head_hidden,
            num_layers=head_layers,
            dropout=head_dropout if head_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Linear(head_hidden * 2, num_labels)
        self.ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    def forward(
        self,
        audio: torch.Tensor,
        audio_len: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        label_len: Optional[torch.Tensor] = None,
    ):
        # Convert batched waveform to Whisper log-mel input features
        # WhisperFeatureExtractor expects 16k mono audio.
        audio_np = audio.detach().cpu().numpy()
        feats = self.feature_extractor(
            audio_np,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to(audio.device)

        enc_out = self.encoder(input_features=feats, output_hidden_states=True)
        rep = enc_out.last_hidden_state  # (B, T', D)
        if self.training and self.use_specaugment:
            rep = apply_specaugment(rep)

        rep, _ = self.head(rep)
        logits = self.classifier(rep)
        out_len = torch.full((audio.size(0),), logits.size(1), dtype=torch.long, device=audio.device)

        output = {"logits": logits, "output_len": out_len}
        if labels is not None and label_len is not None:
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            loss = self.ctc_loss(log_probs, labels, out_len, label_len)
            output["loss"] = loss
        return output

