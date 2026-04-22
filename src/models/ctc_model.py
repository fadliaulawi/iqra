"""SSL encoder + CTC head model (frozen or full fine-tuning)."""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, Wav2Vec2ForCTC
from src.data.augmentation import apply_specaugment


class LayerWeightedSum(nn.Module):
    """Learned weighted sum of transformer hidden states."""

    def __init__(self, n_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_layers))

    def forward(self, hidden_states):
        # hidden_states includes embedding output + transformer layers
        stacked = torch.stack(hidden_states, dim=0)  # (L, B, T, D)
        alpha = torch.softmax(self.weights, dim=0).view(-1, 1, 1, 1)
        return torch.sum(alpha * stacked, dim=0)  # (B, T, D)


class FrozenSSLCTC(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        num_labels: int,
        blank_id: int,
        head_hidden: int = 1024,
        head_layers: int = 2,
        head_dropout: float = 0.1,
        gradient_checkpointing: bool = False,
        mms_target_lang: Optional[str] = None,
        freeze_encoder: bool = True,
        specaugment: bool = False,
    ):
        super().__init__()
        self.blank_id = blank_id

        # MMS ASR checkpoints (e.g. facebook/mms-1b-all): adapters + lm_head live on Wav2Vec2ForCTC.
        # AutoModel -> Wav2Vec2Model has no `target_lang`; load_adapter() then crashes.
        # Official pattern: Wav2Vec2ForCTC.from_pretrained(..., target_lang=...) then use .wav2vec2.
        if mms_target_lang:
            _ctc = Wav2Vec2ForCTC.from_pretrained(
                encoder_name,
                target_lang=mms_target_lang,
            )
            self.encoder = _ctc.wav2vec2
            del _ctc
        else:
            self.encoder = AutoModel.from_pretrained(encoder_name)

        self.freeze_encoder = freeze_encoder
        self.use_specaugment = specaugment

        for p in self.encoder.parameters():
            p.requires_grad = not self.freeze_encoder

        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

        hidden_size = int(self.encoder.config.hidden_size)
        # Number of hidden states returned is usually num_hidden_layers + 1
        n_states = int(getattr(self.encoder.config, "num_hidden_layers", 12)) + 1
        self.layer_sum = LayerWeightedSum(n_states)

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

    def _feat_lengths(self, audio_len: torch.Tensor, time_steps: int) -> torch.Tensor:
        if hasattr(self.encoder, "_get_feat_extract_output_lengths"):
            lengths = self.encoder._get_feat_extract_output_lengths(audio_len)
            return lengths.clamp(max=time_steps)
        # Fallback: assume all same as output time dimension.
        return torch.full_like(audio_len, fill_value=time_steps)

    def forward(
        self,
        audio: torch.Tensor,
        audio_len: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        label_len: Optional[torch.Tensor] = None,
    ):
        # audio: (B, T)
        attn = (torch.arange(audio.size(1), device=audio.device)[None, :] < audio_len[:, None]).long()

        enc_out = self.encoder(
            input_values=audio,
            attention_mask=attn,
            output_hidden_states=True,
        )
        rep = self.layer_sum(enc_out.hidden_states)
        if self.training and self.use_specaugment:
            rep = apply_specaugment(rep)
        rep, _ = self.head(rep)
        logits = self.classifier(rep)  # (B, T', V)
        out_len = self._feat_lengths(audio_len, logits.size(1))

        output = {"logits": logits, "output_len": out_len}
        if labels is not None and label_len is not None:
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # (T', B, V)
            loss = self.ctc_loss(log_probs, labels, out_len, label_len)
            output["loss"] = loss
        return output

