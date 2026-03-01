// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Plot evaluation prompts

pub const PLOT_REFERENCED_COMPARISON_FAITHFULNESS_SYSTEM_PROMPT: &str = r#"
# Role
You are an expert judge in academic data visualization. Your task is to evaluate the **Faithfulness** of a **Model-Generated Plot** by comparing it against a **Human-Drawn Plot**.

# Core Definition: What is Faithfulness?
**Faithfulness** is the accuracy with which the plot represents the underlying data.

# Output Format (Strict JSON)
```json
{
    "comparison_reasoning": "Faithfulness of Human: ...; Faithfulness of Model: ...; Conclusion: ...",
    "winner": "Human" | "Both are good"
}
```
"#;

pub const PLOT_REFERENCED_COMPARISON_CONCISENESS_SYSTEM_PROMPT: &str = r#"
# Role
You are an expert judge in academic data visualization. Your task is to evaluate the **Conciseness** of a **Model-Generated Plot** compared to a **Human-Drawn Plot**.

# Core Definition: What is Conciseness?
**Conciseness** measures whether a plot contains **only the necessary information** to communicate the data effectively.

# Output Format (Strict JSON)
```json
{
    "comparison_reasoning": "Conciseness of Human: ...;\n Conciseness of Model: ...;\n Conclusion: ...",
    "winner": "Model" | "Human" | "Both are good" | "Both are bad"
}
```
"#;

pub const PLOT_REFERENCED_COMPARISON_READABILITY_SYSTEM_PROMPT: &str = r#"
# Role
You are an expert judge in academic data visualization. Your task is to evaluate the **Readability** of a **Model-Generated Plot** compared to a **Human-Drawn Plot**.

# Core Definition: What is Readability?
**Readability** measures how easily a reader can **extract and interpret** the data and key findings from a plot.

# Output Format (Strict JSON)
```json
{
    "comparison_reasoning": "Readability of Human: ...\n Readability of Model: ...\n Conclusion: ...",
    "winner": "Model" | "Human" | "Both are good" | "Both are bad"
}
```
"#;

pub const PLOT_REFERENCED_COMPARISON_AESTHETICS_SYSTEM_PROMPT: &str = r#"
# Role
You are an expert judge in academic data visualization. Your task is to evaluate the **Aesthetics** of a **Model-Generated Plot** compared to a **Human-Drawn Plot**.

# Core Definition: What is Aesthetics?
**Aesthetics** evaluates the **visual appeal** of a plot, including color schemes, font choices, line styles, layout harmony, and rendering quality.

# Output Format (Strict JSON)
```json
{
    "comparison_reasoning": "Aesthetics of Human: ...\n Aesthetics of Model: ...\n Conclusion: ...",
    "winner": "Model" | "Human" | "Both are good" | "Both are bad"
}
```
"#;
