"""Local inference utilities for the fine-tuned DeBERTa-v3 classifier."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "saved_models" / "deberta_model"
LABELS = {0: "Non-Disaster", 1: "Disaster"}


@dataclass(frozen=True)
class DisasterPrediction:
    """Prediction returned by the DeBERTa-v3 classifier."""

    label: str
    is_disaster: bool
    confidence: float
    probabilities: dict[str, float]


class DebertaDisasterClassifier:
    """Load the local DeBERTa-v3 model and classify preprocessed tweets."""

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        max_length: int = 128,
        device: str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> DisasterPrediction:
        """Classify one preprocessed tweet."""
        self._validate_text(text)
        probabilities = self._predict_probabilities([text])[0]
        predicted_label = int(torch.argmax(probabilities).item())

        return self._build_prediction(probabilities, predicted_label)

    def predict_batch(self, texts: list[str]) -> list[DisasterPrediction]:
        """Classify a batch of preprocessed tweets."""
        if not texts:
            return []

        for text in texts:
            self._validate_text(text)

        probabilities_batch = self._predict_probabilities(texts)
        return [
            self._build_prediction(
                probabilities,
                int(torch.argmax(probabilities).item()),
            )
            for probabilities in probabilities_batch
        ]

    def _predict_probabilities(self, texts: list[str]) -> torch.Tensor:
        encoded_inputs: dict[str, Any] = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded_inputs = {
            name: tensor.to(self.device) for name, tensor in encoded_inputs.items()
        }

        with torch.inference_mode():
            logits = self.model(**encoded_inputs).logits

        return torch.softmax(logits, dim=-1).cpu()

    @staticmethod
    def _build_prediction(
        probabilities: torch.Tensor,
        predicted_label: int,
    ) -> DisasterPrediction:
        return DisasterPrediction(
            label=LABELS[predicted_label],
            is_disaster=predicted_label == 1,
            confidence=float(probabilities[predicted_label].item()),
            probabilities={
                LABELS[index]: float(probability.item())
                for index, probability in enumerate(probabilities)
            },
        )

    @staticmethod
    def _validate_text(text: str) -> None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string.")
