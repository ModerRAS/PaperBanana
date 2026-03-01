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
Tests for get_config_val utility function.
"""

import os
import pytest
from unittest.mock import patch


class TestGetConfigVal:
    """Tests for get_config_val."""

    def _get_fn(self):
        from utils.generation_utils import get_config_val
        return get_config_val

    def test_env_var_takes_precedence(self):
        fn = self._get_fn()
        with patch.dict(os.environ, {"TEST_KEY": "from_env"}):
            result = fn("api_keys", "test_key", "TEST_KEY", "default_val")
        assert result == "from_env"

    def test_default_when_no_env_or_config(self):
        fn = self._get_fn()
        with patch.dict(os.environ, {}, clear=False):
            # Ensure the env var isn't set
            os.environ.pop("NONEXISTENT_KEY_12345", None)
            result = fn("nonexistent_section", "nonexistent_key", "NONEXISTENT_KEY_12345", "my_default")
        assert result == "my_default"

    def test_empty_default(self):
        fn = self._get_fn()
        os.environ.pop("NONEXISTENT_KEY_67890", None)
        result = fn("no_section", "no_key", "NONEXISTENT_KEY_67890")
        assert result == ""
