#!/bin/bash
# Copyright 2026 Google LLC
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

set -e

# Resolve Project Root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Create necessary data directories if they don't exist
mkdir -p data/PaperBananaBench/diagram
mkdir -p data/PaperBananaBench/plot
if [ ! -f "data/PaperBananaBench/diagram/ref.json" ]; then
    echo "[]" > data/PaperBananaBench/diagram/ref.json
fi
if [ ! -f "data/PaperBananaBench/plot/ref.json" ]; then
    echo "[]" > data/PaperBananaBench/plot/ref.json
fi

# Build and run the CLI demo
echo "Building paper_banana_demo..."
cargo build --release --bin paper_banana_demo

echo ""
echo "🍌 PaperBanana Demo"
echo "Usage:"
echo "  Generate candidates:"
echo "    cargo run --release --bin paper_banana_demo -- \\"
echo "      --method '<method section text>' \\"
echo "      --caption '<figure caption>' \\"
echo "      --num-candidates 3 \\"
echo "      --exp-mode dev_full"
echo ""
echo "  Refine an image:"
echo "    cargo run --release --bin paper_banana_demo -- refine \\"
echo "      --image-path path/to/image.jpg \\"
echo "      --instructions 'Make the text larger'"
echo ""

# Run demo with sample input if no args provided
if [ "$#" -eq 0 ]; then
    echo "Running demo with sample input..."
    cargo run --release --bin paper_banana_demo -- \
        --method "We propose a multi-agent framework with five specialized components: Retriever, Planner, Stylist, Visualizer and Critic." \
        --caption "Overview of our proposed multi-agent pipeline for academic illustration generation." \
        --num-candidates 1 \
        --exp-mode "demo_full" \
        --retrieval-setting "none"
else
    cargo run --release --bin paper_banana_demo -- "$@"
fi
