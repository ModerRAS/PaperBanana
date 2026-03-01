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
API compatibility tests for Doubao/Volcengine Ark API.

These tests verify that our implementation matches the official Volcengine Ark
Python SDK source code (volcenginesdkarkruntime) to ensure there are no
interface-level incompatibilities.

Reference source of truth:
  - Chat: volcenginesdkarkruntime/resources/chat/completions.py
  - Images: volcenginesdkarkruntime/resources/images/images.py

Since PaperBanana uses the OpenAI Python SDK (AsyncOpenAI) with a custom
base_url pointing to the Ark platform, non-standard Ark params like
guidance_scale/watermark must go through extra_body.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock


class TestDoubaoLLMApiCompatibility:
    """
    Verify call_doubao_with_retry_async matches the Volcengine Ark
    Chat Completions API specification.

    Ark API reference: POST {base_url}/chat/completions
    - model: string (required)
    - messages: array of {role, content} (required)
    - temperature: float (optional, 0-2, default 1)
    - max_tokens: integer (optional, default 4096)
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        from utils.generation_utils import call_doubao_with_retry_async
        self.call_fn = call_doubao_with_retry_async

        self.mock_response = MagicMock()
        self.mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]

    @pytest.mark.asyncio
    async def test_ark_uses_max_tokens_not_max_completion_tokens(self):
        """
        Ark API docs specify 'max_tokens' (not 'max_completion_tokens').
        This is different from newer OpenAI API which uses 'max_completion_tokens'.
        """
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=self.mock_response)

        with patch("utils.generation_utils.doubao_client", mock_client):
            await self.call_fn(
                model_name="doubao-pro-32k",
                contents=[{"type": "text", "text": "test"}],
                config={
                    "system_prompt": "sys",
                    "temperature": 1.0,
                    "candidate_num": 1,
                    "max_output_tokens": 4096,
                },
                max_attempts=1,
            )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "max_tokens" in call_kwargs, "Ark API uses 'max_tokens', not 'max_completion_tokens'"
        assert "max_completion_tokens" not in call_kwargs

    @pytest.mark.asyncio
    async def test_ark_messages_format_system_and_user(self):
        """
        Ark API supports standard OpenAI messages format:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        """
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=self.mock_response)

        with patch("utils.generation_utils.doubao_client", mock_client):
            await self.call_fn(
                model_name="doubao-pro-32k",
                contents=[{"type": "text", "text": "hello"}],
                config={
                    "system_prompt": "You are helpful",
                    "temperature": 0.7,
                    "candidate_num": 1,
                    "max_output_tokens": 4096,
                },
                max_attempts=1,
            )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful"
        assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_ark_temperature_passed_correctly(self):
        """Ark API temperature param should be passed directly."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=self.mock_response)

        with patch("utils.generation_utils.doubao_client", mock_client):
            await self.call_fn(
                model_name="doubao-pro-32k",
                contents=[{"type": "text", "text": "hello"}],
                config={
                    "system_prompt": "sys",
                    "temperature": 0.3,
                    "candidate_num": 1,
                    "max_output_tokens": 4096,
                },
                max_attempts=1,
            )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_ark_model_name_passed_directly(self):
        """Ark API model param should be the endpoint ID or model name."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=self.mock_response)

        with patch("utils.generation_utils.doubao_client", mock_client):
            await self.call_fn(
                model_name="doubao-pro-128k",
                contents=[{"type": "text", "text": "hello"}],
                config={
                    "system_prompt": "sys",
                    "temperature": 1.0,
                    "candidate_num": 1,
                    "max_output_tokens": 4096,
                },
                max_attempts=1,
            )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "doubao-pro-128k"

    @pytest.mark.asyncio
    async def test_ark_vision_format_uses_image_url(self):
        """
        Ark vision API (对话-视觉) uses same format as OpenAI:
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        """
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=self.mock_response)

        contents = [
            {"type": "text", "text": "Describe this image"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": "abc123",
                },
            },
        ]

        with patch("utils.generation_utils.doubao_client", mock_client):
            await self.call_fn(
                model_name="doubao-vision-pro",
                contents=contents,
                config={
                    "system_prompt": "sys",
                    "temperature": 1.0,
                    "candidate_num": 1,
                    "max_output_tokens": 4096,
                },
                max_attempts=1,
            )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        user_content = call_kwargs["messages"][1]["content"]
        # Should have converted to OpenAI image_url format
        assert any(item["type"] == "image_url" for item in user_content)
        img_item = [item for item in user_content if item["type"] == "image_url"][0]
        assert img_item["image_url"]["url"].startswith("data:image/jpeg;base64,")


class TestDoubaoImageApiCompatibility:
    """
    Verify call_doubao_image_generation_with_retry_async matches the Volcengine
    Ark Images Generations API specification.

    Ark API reference: POST {base_url}/images/generations
    Top-level params: model, prompt, n, size, response_format, seed,
                      guidance_scale, watermark
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        from utils.generation_utils import call_doubao_image_generation_with_retry_async
        self.call_fn = call_doubao_image_generation_with_retry_async

        self.mock_img_data = MagicMock()
        self.mock_img_data.b64_json = "test_b64_data"
        self.mock_response = MagicMock()
        self.mock_response.data = [self.mock_img_data]

    @pytest.mark.asyncio
    async def test_ark_native_sdk_no_n_param(self):
        """
        Volcengine SDK images.generate() does NOT have an 'n' parameter
        (verified from volcenginesdkarkruntime source). Since we now use the
        native SDK, 'n' should NOT be in the call params.
        """
        mock_client = AsyncMock()
        mock_client.images.generate = AsyncMock(return_value=self.mock_response)

        with patch("utils.generation_utils.doubao_client", mock_client):
            await self.call_fn(
                model_name="doubao-seedream-3-0-t2i",
                prompt="A cat",
                config={"size": "1024x1024"},
                max_attempts=1,
            )
        call_kwargs = mock_client.images.generate.call_args[1]
        # Native Volcengine SDK does not use 'n' parameter
        assert "n" not in call_kwargs

    @pytest.mark.asyncio
    async def test_ark_image_guidance_scale_as_direct_param(self):
        """
        With native Volcengine SDK, guidance_scale should be a direct parameter
        (not via extra_body). Verified from volcenginesdkarkruntime source:
        images.generate(..., guidance_scale=float, watermark=bool)
        """
        mock_client = AsyncMock()
        mock_client.images.generate = AsyncMock(return_value=self.mock_response)

        with patch("utils.generation_utils.doubao_client", mock_client):
            await self.call_fn(
                model_name="doubao-seedream-3-0-t2i",
                prompt="A sunset",
                config={"guidance_scale": 3.0, "watermark": False},
                max_attempts=1,
            )
        call_kwargs = mock_client.images.generate.call_args[1]
        # Native SDK: guidance_scale is a direct parameter
        assert call_kwargs.get("guidance_scale") == 3.0
        assert call_kwargs.get("watermark") is False

    @pytest.mark.asyncio
    async def test_ark_image_response_format_b64_json(self):
        """Ark API supports response_format='b64_json' for base64 image output."""
        mock_client = AsyncMock()
        mock_client.images.generate = AsyncMock(return_value=self.mock_response)

        with patch("utils.generation_utils.doubao_client", mock_client):
            await self.call_fn(
                model_name="doubao-seedream-3-0-t2i",
                prompt="A forest",
                config={"response_format": "b64_json"},
                max_attempts=1,
            )
        call_kwargs = mock_client.images.generate.call_args[1]
        assert call_kwargs["response_format"] == "b64_json"

    @pytest.mark.asyncio
    async def test_ark_image_size_param(self):
        """Ark API supports size param like '1024x1024', '2K', etc."""
        mock_client = AsyncMock()
        mock_client.images.generate = AsyncMock(return_value=self.mock_response)

        with patch("utils.generation_utils.doubao_client", mock_client):
            await self.call_fn(
                model_name="doubao-seedream-3-0-t2i",
                prompt="A mountain",
                config={"size": "2048x2048"},
                max_attempts=1,
            )
        call_kwargs = mock_client.images.generate.call_args[1]
        assert call_kwargs["size"] == "2048x2048"

    @pytest.mark.asyncio
    async def test_ark_image_model_passed_directly(self):
        """Model name/endpoint ID should be passed directly."""
        mock_client = AsyncMock()
        mock_client.images.generate = AsyncMock(return_value=self.mock_response)

        with patch("utils.generation_utils.doubao_client", mock_client):
            await self.call_fn(
                model_name="doubao-seedream-4-0-250828",
                prompt="A cat",
                config={},
                max_attempts=1,
            )
        call_kwargs = mock_client.images.generate.call_args[1]
        assert call_kwargs["model"] == "doubao-seedream-4-0-250828"

    @pytest.mark.asyncio
    async def test_ark_image_default_guidance_scale_matches_docs(self):
        """Ark docs indicate default guidance_scale of 2.5."""
        mock_client = AsyncMock()
        mock_client.images.generate = AsyncMock(return_value=self.mock_response)

        with patch("utils.generation_utils.doubao_client", mock_client):
            await self.call_fn(
                model_name="doubao-seedream-3-0-t2i",
                prompt="A dog",
                config={},  # No guidance_scale specified, should use default
                max_attempts=1,
            )
        call_kwargs = mock_client.images.generate.call_args[1]
        assert call_kwargs.get("guidance_scale") == 2.5

    @pytest.mark.asyncio
    async def test_ark_image_response_format_url_returns_url(self):
        """
        TDD: Ark API supports response_format='url' which returns a URL instead of
        b64_json. When response_format='url', data[0].url contains the image URL.
        Our function should handle this correctly.
        (Based on Volcengine SDK: response_format: str | None)
        """
        mock_img_data = MagicMock()
        mock_img_data.b64_json = None  # No b64 data when format is 'url'
        mock_img_data.url = "https://ark-output.example.com/img123.png"
        mock_response = MagicMock()
        mock_response.data = [mock_img_data]

        mock_client = AsyncMock()
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        with patch("utils.generation_utils.doubao_client", mock_client):
            result = await self.call_fn(
                model_name="doubao-seedream-3-0-t2i",
                prompt="A sunset over mountains",
                config={"response_format": "url"},
                max_attempts=1,
            )
        # When response_format='url', should return the URL
        assert result == ["https://ark-output.example.com/img123.png"]

    @pytest.mark.asyncio
    async def test_ark_native_sdk_no_n_param_in_image_gen(self):
        """
        Volcengine SDK images.generate() does NOT have an 'n' parameter.
        Since we use the native SDK, verify 'n' is not passed.
        """
        mock_client = AsyncMock()
        mock_client.images.generate = AsyncMock(return_value=self.mock_response)

        with patch("utils.generation_utils.doubao_client", mock_client):
            await self.call_fn(
                model_name="doubao-seedream-3-0-t2i",
                prompt="A cat",
                config={},
                max_attempts=1,
            )
        call_kwargs = mock_client.images.generate.call_args[1]
        assert "n" not in call_kwargs


class TestDoubaoLLMSdkCompatibility:
    """
    Additional tests verifying Doubao LLM compatibility based on the official
    Volcengine SDK source (volcenginesdkarkruntime/resources/chat/completions.py).

    The SDK supports BOTH max_tokens and max_completion_tokens.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        from utils.generation_utils import call_doubao_with_retry_async
        self.call_fn = call_doubao_with_retry_async

        self.mock_response = MagicMock()
        self.mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]

    @pytest.mark.asyncio
    async def test_sdk_supports_both_max_tokens_variants(self):
        """
        Official Volcengine SDK has BOTH max_tokens and max_completion_tokens.
        Our code uses max_tokens which is the standard OpenAI-compatible name.
        Verify this doesn't send max_completion_tokens.
        """
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=self.mock_response)

        with patch("utils.generation_utils.doubao_client", mock_client):
            await self.call_fn(
                model_name="doubao-pro-32k",
                contents=[{"type": "text", "text": "test"}],
                config={
                    "system_prompt": "sys",
                    "temperature": 1.0,
                    "candidate_num": 1,
                    "max_output_tokens": 4096,
                },
                max_attempts=1,
            )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        # Should use max_tokens (not max_completion_tokens)
        assert "max_tokens" in call_kwargs
        assert call_kwargs["max_tokens"] == 4096
        # Should NOT also send max_completion_tokens (avoid conflict)
        assert "max_completion_tokens" not in call_kwargs
