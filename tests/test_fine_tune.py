"""Tests for fine-tuning pipeline (data loading, trainer invocation, load_model).

The fine_tune module uses lazy imports for datasets and sentence-transformers
training API. Tests mock at the import boundary to avoid requiring heavy
GPU dependencies.
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from src.embed import load_model


class TestLoadModelRevision:
    @patch("src.embed.SentenceTransformer")
    def test_revision_null_omits_kwarg(self, mock_st):
        """When revision is null (None), revision kwarg should not be passed."""
        config = {"id": "models/bge-large-finetuned", "revision": None}
        load_model(config)

        mock_st.assert_called_once_with(
            "models/bge-large-finetuned",
            trust_remote_code=False,
        )

    @patch("src.embed.SentenceTransformer")
    def test_revision_present_passes_kwarg(self, mock_st):
        """When revision is set, it should be passed to SentenceTransformer."""
        config = {
            "id": "BAAI/bge-large-en-v1.5",
            "revision": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        }
        load_model(config)

        mock_st.assert_called_once_with(
            "BAAI/bge-large-en-v1.5",
            trust_remote_code=False,
            revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        )

    @patch("src.embed.SentenceTransformer")
    def test_revision_missing_key_omits_kwarg(self, mock_st):
        """When revision key is absent entirely, revision kwarg should not be passed."""
        config = {"id": "sentence-transformers/all-MiniLM-L6-v2"}
        load_model(config)

        mock_st.assert_called_once_with(
            "sentence-transformers/all-MiniLM-L6-v2",
            trust_remote_code=False,
        )


class TestTsdaeTrainer:
    def test_tsdae_trainer_called(self, tmp_path):
        """train_tsdae should build a Dataset, create TSDAE loss, and train."""
        # Build mock modules
        mock_dataset_cls = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.__len__ = lambda self: 2
        mock_dataset_cls.from_dict.return_value = mock_dataset

        mock_datasets_mod = MagicMock()
        mock_datasets_mod.Dataset = mock_dataset_cls

        mock_model = MagicMock()
        mock_st_mod = MagicMock()
        mock_st_mod.SentenceTransformer.return_value = mock_model

        mock_loss = MagicMock()
        mock_st_mod.losses.DenoisingAutoEncoderLoss.return_value = mock_loss

        mock_trainer = MagicMock()
        mock_st_mod.SentenceTransformerTrainer.return_value = mock_trainer

        mock_args_cls = MagicMock()
        mock_args = MagicMock()
        mock_args_cls.return_value = mock_args

        mock_args_mod = MagicMock()
        mock_args_mod.SentenceTransformerTrainingArguments = mock_args_cls

        corpus = tmp_path / "corpus.txt"
        corpus.write_text("Software Engineer\nData Analyst\n")
        output = str(tmp_path / "tsdae-out")

        with patch.dict(sys.modules, {
            "datasets": mock_datasets_mod,
            "sentence_transformers": mock_st_mod,
            "sentence_transformers.training_args": mock_args_mod,
        }):
            from src.fine_tune import train_tsdae
            result = train_tsdae("BAAI/bge-large-en-v1.5", str(corpus), output)

        # Verify Dataset was created with sentence column
        call_args = mock_dataset_cls.from_dict.call_args
        assert "sentence" in call_args[0][0]
        assert len(call_args[0][0]["sentence"]) == 2

        # Verify TSDAE loss
        mock_st_mod.losses.DenoisingAutoEncoderLoss.assert_called_once()
        loss_kwargs = mock_st_mod.losses.DenoisingAutoEncoderLoss.call_args
        assert loss_kwargs[1]["tie_encoder_decoder"] is True

        # Verify trainer was invoked
        mock_trainer.train.assert_called_once()
        mock_model.save.assert_called_once_with(output)
        assert result == output


class TestContrastiveTrainer:
    def test_contrastive_trainer_called(self, tmp_path):
        """train_contrastive should build triplet Dataset, create MNR loss, and train."""
        mock_dataset_cls = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.__len__ = lambda self: 2
        mock_dataset.column_names = ["anchor", "positive", "negative"]
        mock_dataset_cls.from_dict.return_value = mock_dataset

        mock_datasets_mod = MagicMock()
        mock_datasets_mod.Dataset = mock_dataset_cls

        mock_model = MagicMock()
        mock_st_mod = MagicMock()
        mock_st_mod.SentenceTransformer.return_value = mock_model

        mock_loss = MagicMock()
        mock_st_mod.losses.MultipleNegativesRankingLoss.return_value = mock_loss

        mock_trainer = MagicMock()
        mock_st_mod.SentenceTransformerTrainer.return_value = mock_trainer

        mock_args_cls = MagicMock()
        mock_args_mod = MagicMock()
        mock_args_mod.SentenceTransformerTrainingArguments = mock_args_cls

        pairs = tmp_path / "pairs.jsonl"
        triplets = [
            {"anchor": "Software Engineer", "positive": "Sr Eng", "negative": "Sales Rep"},
            {"anchor": "Data Analyst", "positive": "Anlst", "negative": "PM"},
        ]
        pairs.write_text("\n".join(json.dumps(t) for t in triplets) + "\n")
        output = str(tmp_path / "ft-out")

        with patch.dict(sys.modules, {
            "datasets": mock_datasets_mod,
            "sentence_transformers": mock_st_mod,
            "sentence_transformers.training_args": mock_args_mod,
        }):
            from src.fine_tune import train_contrastive
            result = train_contrastive(str(tmp_path / "model"), str(pairs), output)

        # Verify Dataset columns
        call_args = mock_dataset_cls.from_dict.call_args
        data_dict = call_args[0][0]
        assert set(data_dict.keys()) == {"anchor", "positive", "negative"}
        assert len(data_dict["anchor"]) == 2

        # Verify MNR loss
        mock_st_mod.losses.MultipleNegativesRankingLoss.assert_called_once_with(mock_model)

        # Verify trainer
        mock_trainer.train.assert_called_once()
        mock_model.save.assert_called_once_with(output)
        assert result == output
