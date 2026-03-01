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
Tests for call_text_model_with_retry_async dispatcher routing.
"""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock


class TestTextModelDispatcherRouting:
    """Tests that call_text_model_with_retry_async routes to the correct provider."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from utils.generation_utils import call_text_model_with_retry_async
        self.dispatch = call_text_model_with_retry_async

    @pytest.mark.asyncio
    async def test_gemini_routing(self):
        with patch("utils.generation_utils.call_gemini_with_retry_async", new_callable=AsyncMock) as mock:
            mock.return_value = ["gemini response"]
            result = await self.dispatch(
                model_name="gemini-3-pro",
                contents=[{"type": "text", "text": "hi"}],
                system_prompt="sys",
                temperature=0.5,
            )
            mock.assert_called_once()
            assert result == ["gemini response"]

    @pytest.mark.asyncio
    async def test_doubao_routing(self):
        with patch("utils.generation_utils.call_doubao_with_retry_async", new_callable=AsyncMock) as mock:
            mock.return_value = ["doubao response"]
            result = await self.dispatch(
                model_name="doubao-pro-32k",
                contents=[{"type": "text", "text": "hi"}],
                system_prompt="sys",
                temperature=0.5,
            )
            mock.assert_called_once()
            # Verify config structure passed to doubao
            config_arg = mock.call_args[1]["config"]
            assert "system_prompt" in config_arg
            assert "temperature" in config_arg
            assert "candidate_num" in config_arg
            assert "max_output_tokens" in config_arg
            assert result == ["doubao response"]

    @pytest.mark.asyncio
    async def test_claude_routing(self):
        with patch("utils.generation_utils.call_claude_with_retry_async", new_callable=AsyncMock) as mock:
            mock.return_value = ["claude response"]
            result = await self.dispatch(
                model_name="claude-3.5-sonnet",
                contents=[{"type": "text", "text": "hi"}],
                system_prompt="sys",
                temperature=0.5,
            )
            mock.assert_called_once()
            config_arg = mock.call_args[1]["config"]
            assert "system_prompt" in config_arg
            assert "max_output_tokens" in config_arg
            assert result == ["claude response"]

    @pytest.mark.asyncio
    async def test_openai_gpt_routing(self):
        with patch("utils.generation_utils.call_openai_with_retry_async", new_callable=AsyncMock) as mock:
            mock.return_value = ["gpt response"]
            result = await self.dispatch(
                model_name="gpt-4o",
                contents=[{"type": "text", "text": "hi"}],
                system_prompt="sys",
                temperature=0.5,
            )
            mock.assert_called_once()
            config_arg = mock.call_args[1]["config"]
            assert "max_completion_tokens" in config_arg
            assert result == ["gpt response"]

    @pytest.mark.asyncio
    async def test_openai_o1_routing(self):
        with patch("utils.generation_utils.call_openai_with_retry_async", new_callable=AsyncMock) as mock:
            mock.return_value = ["o1 response"]
            result = await self.dispatch(
                model_name="o1-preview",
                contents=[{"type": "text", "text": "hi"}],
                system_prompt="sys",
                temperature=0.5,
            )
            mock.assert_called_once()
            assert result == ["o1 response"]

    @pytest.mark.asyncio
    async def test_openai_o3_routing(self):
        with patch("utils.generation_utils.call_openai_with_retry_async", new_callable=AsyncMock) as mock:
            mock.return_value = ["o3 response"]
            result = await self.dispatch(
                model_name="o3-mini",
                contents=[{"type": "text", "text": "hi"}],
                system_prompt="sys",
                temperature=0.5,
            )
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_openai_o4_routing(self):
        with patch("utils.generation_utils.call_openai_with_retry_async", new_callable=AsyncMock) as mock:
            mock.return_value = ["o4 response"]
            result = await self.dispatch(
                model_name="o4-mini",
                contents=[{"type": "text", "text": "hi"}],
                system_prompt="sys",
                temperature=0.5,
            )
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_unsupported_model_raises(self):
        with pytest.raises(ValueError, match="Unsupported text model"):
            await self.dispatch(
                model_name="unknown-model-xyz",
                contents=[{"type": "text", "text": "hi"}],
                system_prompt="sys",
                temperature=0.5,
            )

    @pytest.mark.asyncio
    async def test_doubao_config_keys_match_handler(self):
        """Verify the config keys the dispatcher passes match what call_doubao_with_retry_async expects."""
        with patch("utils.generation_utils.call_doubao_with_retry_async", new_callable=AsyncMock) as mock:
            mock.return_value = ["ok"]
            await self.dispatch(
                model_name="doubao-lite",
                contents=[{"type": "text", "text": "hi"}],
                system_prompt="test_sys",
                temperature=0.7,
                candidate_num=2,
                max_output_tokens=8192,
            )
            config = mock.call_args[1]["config"]
            assert config["system_prompt"] == "test_sys"
            assert config["temperature"] == 0.7
            assert config["candidate_num"] == 2
            assert config["max_output_tokens"] == 8192

    @pytest.mark.asyncio
    async def test_openai_config_keys_match_handler(self):
        """Verify the config keys the dispatcher passes match what call_openai_with_retry_async expects."""
        with patch("utils.generation_utils.call_openai_with_retry_async", new_callable=AsyncMock) as mock:
            mock.return_value = ["ok"]
            await self.dispatch(
                model_name="gpt-4o",
                contents=[{"type": "text", "text": "hi"}],
                system_prompt="test_sys",
                temperature=0.7,
                candidate_num=2,
                max_output_tokens=8192,
            )
            config = mock.call_args[1]["config"]
            # OpenAI uses max_completion_tokens, not max_output_tokens
            assert "max_completion_tokens" in config
            assert config["max_completion_tokens"] == 8192
            assert config["system_prompt"] == "test_sys"

    @pytest.mark.asyncio
    async def test_claude_config_keys_match_handler(self):
        """Verify the config keys the dispatcher passes match what call_claude_with_retry_async expects."""
        with patch("utils.generation_utils.call_claude_with_retry_async", new_callable=AsyncMock) as mock:
            mock.return_value = ["ok"]
            await self.dispatch(
                model_name="claude-3-opus",
                contents=[{"type": "text", "text": "hi"}],
                system_prompt="test_sys",
                temperature=0.7,
                candidate_num=2,
                max_output_tokens=8192,
            )
            config = mock.call_args[1]["config"]
            # Claude uses max_output_tokens
            assert "max_output_tokens" in config
            assert config["max_output_tokens"] == 8192
