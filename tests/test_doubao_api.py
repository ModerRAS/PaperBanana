# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for Doubao API functions (LLM, Image generation).
"""

import asyncio
import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock, PropertyMock


class TestDoubaoLLMRetryAsync:
    """Tests for call_doubao_with_retry_async."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from utils.generation_utils import call_doubao_with_retry_async
        self.call_fn = call_doubao_with_retry_async

    @pytest.mark.asyncio
    async def test_raises_when_client_is_none(self):
        """Should raise RuntimeError when doubao_client is None."""
        with patch("utils.generation_utils.doubao_client", None):
            with pytest.raises(RuntimeError, match="missing Doubao API key"):
                await self.call_fn(
                    model_name="doubao-pro",
                    contents=[{"type": "text", "text": "hi"}],
                    config={
                        "system_prompt": "sys",
                        "temperature": 0.5,
                        "candidate_num": 1,
                        "max_output_tokens": 4096,
                    },
                )

    @pytest.mark.asyncio
    async def test_success_single_candidate(self):
        """Should return a single response on success."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello from Doubao"))]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("utils.generation_utils.doubao_client", mock_client):
            result = await self.call_fn(
                model_name="doubao-pro",
                contents=[{"type": "text", "text": "hi"}],
                config={
                    "system_prompt": "sys",
                    "temperature": 0.5,
                    "candidate_num": 1,
                    "max_output_tokens": 4096,
                },
                max_attempts=1,
            )
        assert result == ["Hello from Doubao"]

    @pytest.mark.asyncio
    async def test_success_multiple_candidates(self):
        """Should generate remaining candidates after the first succeeds."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="response"))]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("utils.generation_utils.doubao_client", mock_client):
            result = await self.call_fn(
                model_name="doubao-pro",
                contents=[{"type": "text", "text": "hi"}],
                config={
                    "system_prompt": "sys",
                    "temperature": 0.5,
                    "candidate_num": 3,
                    "max_output_tokens": 4096,
                },
                max_attempts=1,
            )
        assert len(result) == 3
        assert all(r == "response" for r in result)

    @pytest.mark.asyncio
    async def test_all_attempts_fail_returns_errors(self):
        """Should return ['Error'] * candidate_num when all attempts fail."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API down"))

        with patch("utils.generation_utils.doubao_client", mock_client):
            result = await self.call_fn(
                model_name="doubao-pro",
                contents=[{"type": "text", "text": "hi"}],
                config={
                    "system_prompt": "sys",
                    "temperature": 0.5,
                    "candidate_num": 2,
                    "max_output_tokens": 4096,
                },
                max_attempts=2,
                retry_delay=0,
            )
        assert result == ["Error", "Error"]

    @pytest.mark.asyncio
    async def test_uses_openai_format_conversion(self):
        """Doubao should use _convert_to_openai_format since it's OpenAI-compatible."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("utils.generation_utils.doubao_client", mock_client), \
             patch("utils.generation_utils._convert_to_openai_format") as mock_convert:
            mock_convert.return_value = [{"type": "text", "text": "hi"}]
            await self.call_fn(
                model_name="doubao-pro",
                contents=[{"type": "text", "text": "hi"}],
                config={
                    "system_prompt": "sys",
                    "temperature": 0.5,
                    "candidate_num": 1,
                    "max_output_tokens": 4096,
                },
                max_attempts=1,
            )
            mock_convert.assert_called()

    @pytest.mark.asyncio
    async def test_uses_max_tokens_param(self):
        """Doubao uses max_tokens (not max_completion_tokens) for the API call."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("utils.generation_utils.doubao_client", mock_client):
            await self.call_fn(
                model_name="doubao-pro",
                contents=[{"type": "text", "text": "hi"}],
                config={
                    "system_prompt": "sys",
                    "temperature": 0.5,
                    "candidate_num": 1,
                    "max_output_tokens": 8192,
                },
                max_attempts=1,
            )
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            # Doubao should use max_tokens, not max_completion_tokens
            assert "max_tokens" in call_kwargs
            assert call_kwargs["max_tokens"] == 8192

    @pytest.mark.asyncio
    async def test_config_fallback_max_completion_tokens(self):
        """Should fallback to max_completion_tokens if max_output_tokens is missing."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("utils.generation_utils.doubao_client", mock_client):
            await self.call_fn(
                model_name="doubao-pro",
                contents=[{"type": "text", "text": "hi"}],
                config={
                    "system_prompt": "sys",
                    "temperature": 0.5,
                    "candidate_num": 1,
                    "max_completion_tokens": 2048,
                },
                max_attempts=1,
            )
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["max_tokens"] == 2048


class TestDoubaoImageGenerationRetryAsync:
    """Tests for call_doubao_image_generation_with_retry_async."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from utils.generation_utils import call_doubao_image_generation_with_retry_async
        self.call_fn = call_doubao_image_generation_with_retry_async

    @pytest.mark.asyncio
    async def test_raises_when_client_is_none(self):
        with patch("utils.generation_utils.doubao_client", None):
            with pytest.raises(RuntimeError, match="missing Doubao API key"):
                await self.call_fn(
                    model_name="doubao-seedream-3-0-t2i",
                    prompt="A cat",
                    config={"size": "1024x1024"},
                )

    @pytest.mark.asyncio
    async def test_success_returns_b64(self):
        """Should return a list with base64 image data on success."""
        mock_img_data = MagicMock()
        mock_img_data.b64_json = "base64_image_data_here"
        mock_response = MagicMock()
        mock_response.data = [mock_img_data]

        mock_client = AsyncMock()
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        with patch("utils.generation_utils.doubao_client", mock_client):
            result = await self.call_fn(
                model_name="doubao-seedream-3-0-t2i",
                prompt="A beautiful sunset",
                config={"size": "1024x1024", "response_format": "b64_json"},
                max_attempts=1,
            )
        assert result == ["base64_image_data_here"]

    @pytest.mark.asyncio
    async def test_default_config_values(self):
        """Should use default config values when not provided."""
        mock_img_data = MagicMock()
        mock_img_data.b64_json = "data"
        mock_response = MagicMock()
        mock_response.data = [mock_img_data]

        mock_client = AsyncMock()
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        with patch("utils.generation_utils.doubao_client", mock_client):
            await self.call_fn(
                model_name="doubao-seedream-3-0-t2i",
                prompt="A cat",
                config={},  # Empty config - should use defaults
                max_attempts=1,
            )
            call_kwargs = mock_client.images.generate.call_args[1]
            assert call_kwargs["size"] == "1024x1024"  # default
            assert call_kwargs["response_format"] == "b64_json"  # default

    @pytest.mark.asyncio
    async def test_guidance_params_as_direct_params(self):
        """Should pass guidance_scale and watermark as direct params (native SDK)."""
        mock_img_data = MagicMock()
        mock_img_data.b64_json = "data"
        mock_response = MagicMock()
        mock_response.data = [mock_img_data]

        mock_client = AsyncMock()
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        with patch("utils.generation_utils.doubao_client", mock_client):
            await self.call_fn(
                model_name="doubao-seedream-3-0-t2i",
                prompt="A cat",
                config={"guidance_scale": 3.0, "watermark": True},
                max_attempts=1,
            )
            call_kwargs = mock_client.images.generate.call_args[1]
            assert call_kwargs.get("guidance_scale") == 3.0
            assert call_kwargs.get("watermark") is True

    @pytest.mark.asyncio
    async def test_no_data_retries(self):
        """Should retry when response has no b64_json data."""
        mock_img_empty = MagicMock()
        mock_img_empty.b64_json = None
        mock_img_empty.url = None
        mock_response_empty = MagicMock()
        mock_response_empty.data = [mock_img_empty]

        mock_img_data = MagicMock()
        mock_img_data.b64_json = "final_data"
        mock_response_ok = MagicMock()
        mock_response_ok.data = [mock_img_data]

        mock_client = AsyncMock()
        mock_client.images.generate = AsyncMock(
            side_effect=[mock_response_empty, mock_response_ok]
        )

        with patch("utils.generation_utils.doubao_client", mock_client):
            result = await self.call_fn(
                model_name="doubao-seedream-3-0-t2i",
                prompt="A cat",
                config={},
                max_attempts=2,
                retry_delay=0,
            )
        assert result == ["final_data"]

    @pytest.mark.asyncio
    async def test_all_attempts_fail_returns_error(self):
        mock_client = AsyncMock()
        mock_client.images.generate = AsyncMock(side_effect=Exception("API error"))

        with patch("utils.generation_utils.doubao_client", mock_client):
            result = await self.call_fn(
                model_name="doubao-seedream-3-0-t2i",
                prompt="A cat",
                config={},
                max_attempts=2,
                retry_delay=0,
            )
        assert result == ["Error"]
