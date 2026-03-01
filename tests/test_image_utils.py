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
Tests for image_utils functions.
"""

import base64
import io
import pytest
from PIL import Image


class TestConvertPngB64ToJpgB64:
    """Tests for convert_png_b64_to_jpg_b64."""

    def _get_fn(self):
        from utils.image_utils import convert_png_b64_to_jpg_b64
        return convert_png_b64_to_jpg_b64

    def _make_png_b64(self, width=10, height=10, color="red"):
        """Create a valid PNG base64 string for testing."""
        img = Image.new("RGB", (width, height), color=color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def test_valid_png_converts_to_jpg(self):
        fn = self._get_fn()
        png_b64 = self._make_png_b64()
        result = fn(png_b64)
        assert result is not None
        # Verify it's valid JPEG
        jpg_bytes = base64.b64decode(result)
        img = Image.open(io.BytesIO(jpg_bytes))
        assert img.format == "JPEG"

    def test_rgba_png_converts_correctly(self):
        fn = self._get_fn()
        # Create RGBA (transparent) PNG
        img = Image.new("RGBA", (10, 10), color=(255, 0, 0, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        result = fn(png_b64)
        assert result is not None
        # JPEG doesn't support alpha, so it should be RGB
        jpg_bytes = base64.b64decode(result)
        img = Image.open(io.BytesIO(jpg_bytes))
        assert img.mode == "RGB"

    def test_none_input_returns_none(self):
        fn = self._get_fn()
        result = fn(None)
        assert result is None

    def test_empty_string_returns_none(self):
        fn = self._get_fn()
        result = fn("")
        assert result is None

    def test_short_string_returns_none(self):
        fn = self._get_fn()
        result = fn("abc")
        assert result is None

    def test_invalid_base64_returns_none(self):
        fn = self._get_fn()
        result = fn("not_valid_base64_image_data_that_is_long_enough")
        assert result is None

    def test_preserves_image_dimensions(self):
        fn = self._get_fn()
        png_b64 = self._make_png_b64(width=50, height=30)
        result = fn(png_b64)
        assert result is not None
        jpg_bytes = base64.b64decode(result)
        img = Image.open(io.BytesIO(jpg_bytes))
        assert img.size == (50, 30)
