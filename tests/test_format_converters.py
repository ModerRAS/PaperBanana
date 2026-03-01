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
Tests for format conversion functions in generation_utils.
"""

import base64
import pytest
from unittest.mock import patch, MagicMock

# We need to mock the heavy imports before importing generation_utils
# to avoid requiring API keys and SDK installs at test time.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConvertToOpenAIFormat:
    """Tests for _convert_to_openai_format."""

    def _get_fn(self):
        from utils.generation_utils import _convert_to_openai_format
        return _convert_to_openai_format

    def test_text_only(self):
        fn = self._get_fn()
        contents = [{"type": "text", "text": "Hello world"}]
        result = fn(contents)
        assert result == [{"type": "text", "text": "Hello world"}]

    def test_image_base64(self):
        fn = self._get_fn()
        contents = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "abc123",
                },
            }
        ]
        result = fn(contents)
        assert len(result) == 1
        assert result[0]["type"] == "image_url"
        assert result[0]["image_url"]["url"] == "data:image/png;base64,abc123"

    def test_image_default_media_type(self):
        fn = self._get_fn()
        contents = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "data": "abc123",
                },
            }
        ]
        result = fn(contents)
        # Default media type is image/jpeg
        assert result[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_mixed_content(self):
        fn = self._get_fn()
        contents = [
            {"type": "text", "text": "Look at this image:"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": "imgdata",
                },
            },
            {"type": "text", "text": "What do you see?"},
        ]
        result = fn(contents)
        assert len(result) == 3
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image_url"
        assert result[2]["type"] == "text"

    def test_empty_content(self):
        fn = self._get_fn()
        result = fn([])
        assert result == []

    def test_unknown_type_skipped(self):
        fn = self._get_fn()
        contents = [
            {"type": "video", "url": "http://example.com/video.mp4"},
            {"type": "text", "text": "hello"},
        ]
        result = fn(contents)
        # Only the text item should be converted; unknown types are silently skipped
        assert len(result) == 1
        assert result[0]["type"] == "text"

    def test_image_non_base64_source_skipped(self):
        fn = self._get_fn()
        contents = [
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "http://example.com/img.png",
                },
            }
        ]
        result = fn(contents)
        # Non-base64 image sources are skipped
        assert result == []


class TestConvertToClaudeFormat:
    """Tests for _convert_to_claude_format."""

    def _get_fn(self):
        from utils.generation_utils import _convert_to_claude_format
        return _convert_to_claude_format

    def test_passthrough(self):
        fn = self._get_fn()
        contents = [
            {"type": "text", "text": "hello"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "abc"}},
        ]
        result = fn(contents)
        assert result is contents  # Should be the exact same object (pass-through)


class TestConvertToGeminiParts:
    """Tests for _convert_to_gemini_parts."""

    def _get_fn(self):
        from utils.generation_utils import _convert_to_gemini_parts
        return _convert_to_gemini_parts

    def test_text_part(self):
        fn = self._get_fn()
        contents = [{"type": "text", "text": "Hello Gemini"}]
        result = fn(contents)
        assert len(result) == 1
        # It should be a Gemini Part with text
        assert hasattr(result[0], 'text') or result[0].text == "Hello Gemini"

    def test_image_part(self):
        fn = self._get_fn()
        # Create a minimal valid base64-encoded PNG
        img_data = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("utf-8")
        contents = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_data,
                },
            }
        ]
        result = fn(contents)
        assert len(result) == 1
        # It should be a Gemini Part with inline_data
        assert hasattr(result[0], 'inline_data')

    def test_empty_content(self):
        fn = self._get_fn()
        result = fn([])
        assert result == []

    def test_unknown_type_skipped(self):
        fn = self._get_fn()
        contents = [
            {"type": "audio", "data": "something"},
            {"type": "text", "text": "hello"},
        ]
        result = fn(contents)
        assert len(result) == 1
