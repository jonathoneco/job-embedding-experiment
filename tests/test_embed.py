"""Tests for embedding engine (instruction prefix, reranking, fusion)."""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import yaml

from src.embed import encode_queries, run_embedding_model


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = 384
    # Return normalized vectors by default
    def _encode(texts, batch_size=64, show_progress_bar=False, prompt=None):
        n = len(texts)
        vecs = np.random.default_rng(42).standard_normal((n, 384))
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms
    model.encode.side_effect = _encode
    return model


class TestEncodeQueriesPrompt:
    def test_accepts_prompt_parameter(self, mock_model):
        """encode_queries() should accept prompt kwarg without error."""
        result = encode_queries(
            mock_model, ["test query"], batch_size=32, prompt="instruction: ",
        )
        assert result.shape == (1, 384)

    def test_prompt_none_is_noop(self, mock_model):
        """prompt=None should be backward compatible."""
        result = encode_queries(
            mock_model, ["test query"], batch_size=32, prompt=None,
        )
        assert result.shape == (1, 384)
        # Verify prompt=None was passed to model.encode
        _, kwargs = mock_model.encode.call_args
        assert kwargs["prompt"] is None

    def test_prompt_passed_to_model(self, mock_model):
        """prompt value should be forwarded to model.encode()."""
        encode_queries(
            mock_model,
            ["test query"],
            batch_size=32,
            prompt="Represent this sentence for searching relevant passages: ",
        )
        _, kwargs = mock_model.encode.call_args
        assert kwargs["prompt"] == "Represent this sentence for searching relevant passages: "

    def test_default_prompt_is_none(self, mock_model):
        """Calling without prompt kwarg should default to None."""
        encode_queries(mock_model, ["test query"], batch_size=32)
        _, kwargs = mock_model.encode.call_args
        assert kwargs["prompt"] is None


class TestConfigInstruction:
    @pytest.fixture
    def config(self):
        with open("config.yaml") as f:
            return yaml.safe_load(f)

    def test_bge_base_no_instruction(self, config):
        """BGE instruction prefixing disabled — proven negative for short-text matching."""
        bge_base = [m for m in config["models"] if m["label"] == "bge-base"][0]
        assert "instruction" not in bge_base

    def test_bge_large_no_instruction(self, config):
        """BGE instruction prefixing disabled — proven negative for short-text matching."""
        bge_large = [m for m in config["models"] if m["label"] == "bge-large"][0]
        assert "instruction" not in bge_large

    def test_minilm_has_no_instruction(self, config):
        minilm = [m for m in config["models"] if m["label"] == "minilm"][0]
        assert "instruction" not in minilm


class TestRunEmbeddingModelInstruction:
    @patch("src.embed.load_model")
    @patch("src.embed.encode_targets")
    def test_instruction_threaded_to_encode_queries(
        self, mock_encode_targets, mock_load_model,
    ):
        """run_embedding_model should pass instruction from config to encode_queries."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_load_model.return_value = mock_model

        # Return pre-normalized embeddings
        rng = np.random.default_rng(42)
        target_emb = rng.standard_normal((5, 384)).astype(np.float32)
        target_emb /= np.linalg.norm(target_emb, axis=1, keepdims=True)
        mock_encode_targets.return_value = target_emb

        def _encode(texts, batch_size=64, show_progress_bar=False, prompt=None):
            n = len(texts)
            vecs = rng.standard_normal((n, 384))
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / norms
        mock_model.encode.side_effect = _encode

        model_config = {
            "id": "BAAI/bge-base-en-v1.5",
            "revision": "abc123",
            "label": "bge-base",
            "instruction": "Represent this sentence for searching relevant passages: ",
        }
        target_sets = {
            g: [{"id": f"{g}-{i}", "text": f"text-{i}", "granularity": g}
                for i in range(5)]
            for g in ["role", "role_desc", "cluster", "category_desc", "category"]
        }
        test_cases = [{"id": "tc-1", "input_title": "Dev Mgr"}]
        config = {"embedding": {"batch_size": 64, "cache_dir": "/tmp/test-cache"}}

        run_embedding_model(model_config, target_sets, test_cases, config)

        # Verify prompt was passed in every encode call
        for call in mock_model.encode.call_args_list:
            _, kwargs = call
            assert kwargs["prompt"] == "Represent this sentence for searching relevant passages: "

    @patch("src.embed.load_model")
    @patch("src.embed.encode_targets")
    def test_no_instruction_passes_none(
        self, mock_encode_targets, mock_load_model,
    ):
        """Model config without instruction should pass prompt=None."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_load_model.return_value = mock_model

        rng = np.random.default_rng(42)
        target_emb = rng.standard_normal((5, 384)).astype(np.float32)
        target_emb /= np.linalg.norm(target_emb, axis=1, keepdims=True)
        mock_encode_targets.return_value = target_emb

        def _encode(texts, batch_size=64, show_progress_bar=False, prompt=None):
            n = len(texts)
            vecs = rng.standard_normal((n, 384))
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / norms
        mock_model.encode.side_effect = _encode

        model_config = {
            "id": "sentence-transformers/all-MiniLM-L6-v2",
            "revision": "abc123",
            "label": "minilm",
        }
        target_sets = {
            g: [{"id": f"{g}-{i}", "text": f"text-{i}", "granularity": g}
                for i in range(5)]
            for g in ["role", "role_desc", "cluster", "category_desc", "category"]
        }
        test_cases = [{"id": "tc-1", "input_title": "Software Engineer"}]
        config = {"embedding": {"batch_size": 64, "cache_dir": "/tmp/test-cache"}}

        run_embedding_model(model_config, target_sets, test_cases, config)

        for call in mock_model.encode.call_args_list:
            _, kwargs = call
            assert kwargs["prompt"] is None


def _make_target_sets(granularities, n_targets=5):
    """Helper to build target_sets dict."""
    return {
        g: [{"id": f"{g}-{i}", "text": f"text-{i}", "granularity": g}
            for i in range(n_targets)]
        for g in granularities
    }


def _setup_mock_model(rng=None):
    """Helper to create a mock SentenceTransformer."""
    if rng is None:
        rng = np.random.default_rng(42)
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384

    def _encode(texts, batch_size=64, show_progress_bar=False, prompt=None):
        n = len(texts)
        vecs = rng.standard_normal((n, 384))
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms
    mock_model.encode.side_effect = _encode
    return mock_model


class TestGranularityLoop:
    @patch("src.embed.load_model")
    @patch("src.embed.encode_targets")
    def test_skips_missing_granularity(self, mock_encode_targets, mock_load_model):
        """role_augmented should be skipped when not in target_sets."""
        rng = np.random.default_rng(42)
        mock_model = _setup_mock_model(rng)
        mock_load_model.return_value = mock_model

        target_emb = rng.standard_normal((5, 384)).astype(np.float32)
        target_emb /= np.linalg.norm(target_emb, axis=1, keepdims=True)
        mock_encode_targets.return_value = target_emb

        target_sets = _make_target_sets(
            ["role", "role_desc", "cluster", "category_desc", "category"],
        )
        test_cases = [{"id": "tc-1", "input_title": "Engineer"}]
        config = {"embedding": {"batch_size": 64, "cache_dir": "/tmp/test-cache"}}

        results = run_embedding_model(
            {"id": "test", "revision": "abc", "label": "test"},
            target_sets, test_cases, config,
        )

        granularities = {r["granularity"] for r in results}
        assert "role_augmented" not in granularities
        assert granularities == {"role", "role_desc", "cluster", "category_desc", "category"}

    @patch("src.embed.load_model")
    @patch("src.embed.encode_targets")
    def test_includes_role_augmented_when_present(
        self, mock_encode_targets, mock_load_model,
    ):
        """role_augmented should be included when in target_sets."""
        rng = np.random.default_rng(42)
        mock_model = _setup_mock_model(rng)
        mock_load_model.return_value = mock_model

        target_emb = rng.standard_normal((5, 384)).astype(np.float32)
        target_emb /= np.linalg.norm(target_emb, axis=1, keepdims=True)
        mock_encode_targets.return_value = target_emb

        target_sets = _make_target_sets(
            ["role", "role_desc", "cluster", "category_desc", "category", "role_augmented"],
        )
        test_cases = [{"id": "tc-1", "input_title": "Engineer"}]
        config = {"embedding": {"batch_size": 64, "cache_dir": "/tmp/test-cache"}}

        results = run_embedding_model(
            {"id": "test", "revision": "abc", "label": "test"},
            target_sets, test_cases, config,
        )

        granularities = {r["granularity"] for r in results}
        assert "role_augmented" in granularities


class TestRerankingIntegration:
    @patch("src.embed.load_model")
    @patch("src.embed.encode_targets")
    def test_reranking_disabled_backward_compat(
        self, mock_encode_targets, mock_load_model,
    ):
        """With reranking disabled, no +rerank methods should appear."""
        rng = np.random.default_rng(42)
        mock_model = _setup_mock_model(rng)
        mock_load_model.return_value = mock_model

        target_emb = rng.standard_normal((5, 384)).astype(np.float32)
        target_emb /= np.linalg.norm(target_emb, axis=1, keepdims=True)
        mock_encode_targets.return_value = target_emb

        target_sets = _make_target_sets(
            ["role", "role_desc", "cluster", "category_desc", "category"],
        )
        test_cases = [{"id": "tc-1", "input_title": "Engineer"}]
        config = {
            "embedding": {"batch_size": 64, "cache_dir": "/tmp/test-cache"},
            "reranking": {"enabled": False},
        }

        results = run_embedding_model(
            {"id": "test", "revision": "abc", "label": "test"},
            target_sets, test_cases, config,
        )

        methods = {r["method"] for r in results}
        assert all("+rerank" not in m for m in methods)

    @patch("src.rerank.rerank_batch")
    @patch("src.rerank.load_reranker")
    @patch("src.embed.load_model")
    @patch("src.embed.encode_targets")
    def test_reranking_produces_rerank_results(
        self, mock_encode_targets, mock_load_model,
        mock_load_reranker, mock_rerank_batch,
    ):
        """With reranking enabled, +rerank method entries should be produced."""
        rng = np.random.default_rng(42)
        mock_model = _setup_mock_model(rng)
        mock_load_model.return_value = mock_model

        target_emb = rng.standard_normal((5, 384)).astype(np.float32)
        target_emb /= np.linalg.norm(target_emb, axis=1, keepdims=True)
        mock_encode_targets.return_value = target_emb

        mock_reranker = MagicMock()
        mock_load_reranker.return_value = mock_reranker
        mock_rerank_batch.return_value = [
            [{"target_id": "role-0", "score": 0.95}],
        ]

        target_sets = _make_target_sets(
            ["role", "role_desc", "cluster", "category_desc", "category"],
        )
        test_cases = [{"id": "tc-1", "input_title": "Engineer"}]
        config = {
            "embedding": {"batch_size": 64, "cache_dir": "/tmp/test-cache"},
            "reranking": {
                "enabled": True,
                "model": "cross-encoder/test",
                "initial_k": 5,
                "top_n": 3,
            },
        }

        results = run_embedding_model(
            {"id": "test", "revision": "abc", "label": "test"},
            target_sets, test_cases, config,
        )

        methods = {r["method"] for r in results}
        assert "test+rerank" in methods
        # Should have original + reranked for each of 5 granularities
        reranked = [r for r in results if r["method"] == "test+rerank"]
        assert len(reranked) == 5

    @patch("src.rerank.rerank_batch")
    @patch("src.rerank.load_reranker")
    @patch("src.embed.load_model")
    @patch("src.embed.encode_targets")
    def test_reranking_uses_initial_k(
        self, mock_encode_targets, mock_load_model,
        mock_load_reranker, mock_rerank_batch,
    ):
        """Reranking should pass initial_k to rank_targets for more candidates."""
        rng = np.random.default_rng(42)
        mock_model = _setup_mock_model(rng)
        mock_load_model.return_value = mock_model

        # Need enough targets for initial_k=15
        n_targets = 20
        target_emb = rng.standard_normal((n_targets, 384)).astype(np.float32)
        target_emb /= np.linalg.norm(target_emb, axis=1, keepdims=True)
        mock_encode_targets.return_value = target_emb

        mock_reranker = MagicMock()
        mock_load_reranker.return_value = mock_reranker
        mock_rerank_batch.return_value = [
            [{"target_id": "role-0", "score": 0.95}],
        ]

        target_sets = _make_target_sets(["role"], n_targets=n_targets)
        test_cases = [{"id": "tc-1", "input_title": "Engineer"}]
        config = {
            "embedding": {"batch_size": 64, "cache_dir": "/tmp/test-cache"},
            "reranking": {
                "enabled": True,
                "model": "cross-encoder/test",
                "initial_k": 15,
                "top_n": 5,
            },
        }

        results = run_embedding_model(
            {"id": "test", "revision": "abc", "label": "test"},
            target_sets, test_cases, config,
        )

        # Original results should have initial_k=15 candidates
        original = [r for r in results if r["method"] == "test"]
        assert len(original[0]["ranked_results"]) == 15


class TestFusionIntegration:
    @patch("src.embed.load_model")
    @patch("src.embed.encode_targets")
    def test_fusion_disabled_no_fused_results(
        self, mock_encode_targets, mock_load_model,
    ):
        """With fusion disabled, no fused method entries should appear."""
        rng = np.random.default_rng(42)
        mock_model = _setup_mock_model(rng)
        mock_load_model.return_value = mock_model

        target_emb = rng.standard_normal((5, 384)).astype(np.float32)
        target_emb /= np.linalg.norm(target_emb, axis=1, keepdims=True)
        mock_encode_targets.return_value = target_emb

        target_sets = _make_target_sets(
            ["role", "role_desc", "cluster", "category_desc", "category"],
        )
        test_cases = [{"id": "tc-1", "input_title": "Engineer"}]
        config = {
            "embedding": {"batch_size": 64, "cache_dir": "/tmp/test-cache"},
            "fusion": {"enabled": False, "k": 60, "configs": []},
        }

        results = run_embedding_model(
            {"id": "test", "revision": "abc", "label": "test"},
            target_sets, test_cases, config,
        )

        methods = {r["method"] for r in results}
        assert "test" in methods
        assert len(methods) == 1

    @patch("src.fusion.fuse_all")
    @patch("src.embed.load_model")
    @patch("src.embed.encode_targets")
    def test_fusion_enabled_appends_results(
        self, mock_encode_targets, mock_load_model, mock_fuse_all,
    ):
        """With fusion enabled, fused results should be appended."""
        rng = np.random.default_rng(42)
        mock_model = _setup_mock_model(rng)
        mock_load_model.return_value = mock_model

        target_emb = rng.standard_normal((5, 384)).astype(np.float32)
        target_emb /= np.linalg.norm(target_emb, axis=1, keepdims=True)
        mock_encode_targets.return_value = target_emb

        mock_fuse_all.return_value = [
            {
                "test_case_id": "tc-1",
                "method": "fused-test",
                "granularity": "role",
                "ranked_results": [{"target_id": "role-0", "score": 0.5}],
            },
        ]

        target_sets = _make_target_sets(
            ["role", "role_desc", "cluster", "category_desc", "category"],
        )
        test_cases = [{"id": "tc-1", "input_title": "Engineer"}]
        config = {
            "embedding": {"batch_size": 64, "cache_dir": "/tmp/test-cache"},
            "fusion": {
                "enabled": True,
                "k": 60,
                "configs": [
                    {"name": "fused-test", "methods": ["test"], "granularity": "role"},
                ],
            },
        }

        results = run_embedding_model(
            {"id": "test", "revision": "abc", "label": "test"},
            target_sets, test_cases, config,
        )

        methods = {r["method"] for r in results}
        assert "fused-test" in methods
        mock_fuse_all.assert_called_once()


class TestConfigDefaults:
    @patch("src.embed.load_model")
    @patch("src.embed.encode_targets")
    def test_absent_config_sections(self, mock_encode_targets, mock_load_model):
        """Missing reranking/fusion sections should not cause errors."""
        rng = np.random.default_rng(42)
        mock_model = _setup_mock_model(rng)
        mock_load_model.return_value = mock_model

        target_emb = rng.standard_normal((5, 384)).astype(np.float32)
        target_emb /= np.linalg.norm(target_emb, axis=1, keepdims=True)
        mock_encode_targets.return_value = target_emb

        target_sets = _make_target_sets(
            ["role", "role_desc", "cluster", "category_desc", "category"],
        )
        test_cases = [{"id": "tc-1", "input_title": "Engineer"}]
        # Minimal config — no reranking or fusion sections at all
        config = {"embedding": {"batch_size": 64, "cache_dir": "/tmp/test-cache"}}

        results = run_embedding_model(
            {"id": "test", "revision": "abc", "label": "test"},
            target_sets, test_cases, config,
        )

        assert len(results) == 5  # 5 granularities * 1 test case
        methods = {r["method"] for r in results}
        assert methods == {"test"}
