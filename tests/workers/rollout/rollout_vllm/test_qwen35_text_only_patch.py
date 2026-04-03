# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing as mp
from types import SimpleNamespace

import pytest


def test_patch_vllm_qwen3_5_text_architectures_registers_causal_entries():
    pytest.importorskip("vllm")

    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    from verl.utils.vllm.patch import patch_vllm_qwen3_5_text_architectures

    patch_vllm_qwen3_5_text_architectures()
    patch_vllm_qwen3_5_text_architectures()

    assert "Qwen3_5ForCausalLM" in ModelRegistry.models
    assert "Qwen3_5MoeForCausalLM" in ModelRegistry.models
    assert "Qwen3_5ForCausalLM" in MODELS_CONFIG_MAP
    assert "Qwen3_5MoeForCausalLM" in MODELS_CONFIG_MAP

    stub_model_config = type("StubModelConfig", (), {"runner_type": "generate", "convert_type": "none"})()
    assert ModelRegistry._normalize_arch("Qwen3_5ForCausalLM", stub_model_config) == "Qwen3_5ForCausalLM"
    assert ModelRegistry._normalize_arch("Qwen3_5MoeForCausalLM", stub_model_config) == "Qwen3_5MoeForCausalLM"

    resolved_model_cls, resolved_arch = ModelRegistry.resolve_model_cls(
        ["Qwen3_5ForCausalLM"],
        SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["Qwen3_5ForCausalLM"]),
            model_impl="auto",
            runner_type="generate",
            convert_type="none",
        ),
    )
    assert resolved_arch == "Qwen3_5ForCausalLM"
    assert resolved_model_cls.__name__ == "Qwen3_5ForCausalLM"


def _spawned_worker_registry_probe(queue: mp.Queue):
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    import verl.workers.rollout.vllm_rollout.utils  # noqa: F401

    resolved_model_cls, resolved_arch = ModelRegistry.resolve_model_cls(
        ["Qwen3_5ForCausalLM"],
        SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["Qwen3_5ForCausalLM"]),
            model_impl="auto",
            runner_type="generate",
            convert_type="none",
        ),
    )

    queue.put(
        {
            "registry_has_causal": "Qwen3_5ForCausalLM" in ModelRegistry.models,
            "registry_has_moe_causal": "Qwen3_5MoeForCausalLM" in ModelRegistry.models,
            "config_map_has_causal": "Qwen3_5ForCausalLM" in MODELS_CONFIG_MAP,
            "config_map_has_moe_causal": "Qwen3_5MoeForCausalLM" in MODELS_CONFIG_MAP,
            "resolved_arch": resolved_arch,
            "resolved_model_cls": resolved_model_cls.__name__,
        }
    )


def test_vllm_rollout_utils_patches_qwen3_5_text_architectures_in_spawned_process():
    pytest.importorskip("vllm")

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_spawned_worker_registry_probe, args=(queue,))
    proc.start()
    proc.join(timeout=60)

    assert proc.exitcode == 0
    result = queue.get_nowait()
    assert result["registry_has_causal"]
    assert result["registry_has_moe_causal"]
    assert result["config_map_has_causal"]
    assert result["config_map_has_moe_causal"]
    assert result["resolved_arch"] == "Qwen3_5ForCausalLM"
    assert result["resolved_model_cls"] == "Qwen3_5ForCausalLM"
