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
Tests for OpenAI Image Generation API (call_openai_image_generation_with_retry_async).
"""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock


class TestOpenAIImageGenerationRetryAsync:
    """Tests for call_openai_image_generation_with_retry_async."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from utils.generation_utils import call_openai_image_generation_with_retry_async
        self.call_fn = call_openai_image_generation_with_retry_async

    @pytest.mark.asyncio
    async def test_success_returns_b64(self):
        mock_img_data = MagicMock()
        mock_img_data.b64_json = "openai_b64_data"
        mock_response = MagicMock()
        mock_response.data = [mock_img_data]

        mock_client = AsyncMock()
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        with patch("utils.generation_utils.openai_client", mock_client):
            result = await self.call_fn(
                model_name="gpt-image-1",
                prompt="A sunset",
                config={"size": "1536x1024", "quality": "high"},
                max_attempts=1,
            )
        assert result == ["openai_b64_data"]

    @pytest.mark.asyncio
    async def test_default_config_values(self):
        mock_img_data = MagicMock()
        mock_img_data.b64_json = "data"
        mock_response = MagicMock()
        mock_response.data = [mock_img_data]

        mock_client = AsyncMock()
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        with patch("utils.generation_utils.openai_client", mock_client):
            await self.call_fn(
                model_name="gpt-image-1",
                prompt="A cat",
                config={},  # Empty config
                max_attempts=1,
            )
            call_kwargs = mock_client.images.generate.call_args[1]
            assert call_kwargs["size"] == "1536x1024"  # default
            assert call_kwargs["quality"] == "high"  # default
            assert call_kwargs["background"] == "opaque"  # default
            assert call_kwargs["output_format"] == "png"  # default

    @pytest.mark.asyncio
    async def test_no_b64_data_retries(self):
        mock_img_empty = MagicMock()
        mock_img_empty.b64_json = None
        mock_response_empty = MagicMock()
        mock_response_empty.data = [mock_img_empty]

        mock_img_ok = MagicMock()
        mock_img_ok.b64_json = "final"
        mock_response_ok = MagicMock()
        mock_response_ok.data = [mock_img_ok]

        mock_client = AsyncMock()
        mock_client.images.generate = AsyncMock(
            side_effect=[mock_response_empty, mock_response_ok]
        )

        with patch("utils.generation_utils.openai_client", mock_client):
            result = await self.call_fn(
                model_name="gpt-image-1",
                prompt="A cat",
                config={},
                max_attempts=2,
                retry_delay=0,
            )
        assert result == ["final"]

    @pytest.mark.asyncio
    async def test_all_attempts_fail_returns_error(self):
        mock_client = AsyncMock()
        mock_client.images.generate = AsyncMock(side_effect=Exception("rate limit"))

        with patch("utils.generation_utils.openai_client", mock_client):
            result = await self.call_fn(
                model_name="gpt-image-1",
                prompt="A cat",
                config={},
                max_attempts=2,
                retry_delay=0,
            )
        assert result == ["Error"]

    @pytest.mark.asyncio
    async def test_gpt_image_specific_params(self):
        """OpenAI image gen should include GPT-Image params (quality, background, output_format)."""
        mock_img_data = MagicMock()
        mock_img_data.b64_json = "data"
        mock_response = MagicMock()
        mock_response.data = [mock_img_data]

        mock_client = AsyncMock()
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        with patch("utils.generation_utils.openai_client", mock_client):
            await self.call_fn(
                model_name="gpt-image-1",
                prompt="A cat",
                config={
                    "size": "1024x1024",
                    "quality": "standard",
                    "background": "transparent",
                    "output_format": "webp",
                },
                max_attempts=1,
            )
            call_kwargs = mock_client.images.generate.call_args[1]
            assert call_kwargs["quality"] == "standard"
            assert call_kwargs["background"] == "transparent"
            assert call_kwargs["output_format"] == "webp"
