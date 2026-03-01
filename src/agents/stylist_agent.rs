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

//! Stylist Agent - refines figure descriptions with style guidelines.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;

use crate::agents::base_agent::{Agent, AgentConfig};
use crate::config::ExpConfig;
use crate::generation_utils::{call_text_model_with_retry_async, ApiClients, ContentItem};

struct TaskConfig {
    task_name: String,
    context_labels: [&'static str; 2],
}

pub struct StylistAgent {
    config: AgentConfig,
    task_config: TaskConfig,
}

impl StylistAgent {
    pub fn new(exp_config: ExpConfig) -> Self {
        let (system_prompt, task_config) = if exp_config.task_name == "plot" {
            (
                PLOT_STYLIST_AGENT_SYSTEM_PROMPT.to_string(),
                TaskConfig {
                    task_name: "plot".into(),
                    context_labels: ["Raw Data", "Visual Intent of the Desired Plot"],
                },
            )
        } else {
            (
                DIAGRAM_STYLIST_AGENT_SYSTEM_PROMPT.to_string(),
                TaskConfig {
                    task_name: "diagram".into(),
                    context_labels: ["Methodology Section", "Diagram Caption"],
                },
            )
        };

        let model_name = exp_config.model_name.clone();
        Self {
            config: AgentConfig {
                model_name,
                system_prompt,
                exp_config,
            },
            task_config,
        }
    }
}

#[async_trait]
impl Agent for StylistAgent {
    async fn process(
        &self,
        mut data: HashMap<String, Value>,
        clients: &ApiClients,
    ) -> Result<HashMap<String, Value>> {
        let cfg = &self.task_config;
        let exp = &self.config.exp_config;

        let input_desc_key = format!("target_{}_desc0", cfg.task_name);
        let output_desc_key = format!("target_{}_stylist_desc0", cfg.task_name);

        let detailed_description = data
            .get(&input_desc_key)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let style_guide_path = exp.work_dir.join(format!(
            "style_guides/neurips2025_{}_style_guide.md",
            cfg.task_name
        ));
        let style_guide = fs::read_to_string(&style_guide_path).unwrap_or_else(|e| {
            println!(
                "Warning: Could not read style guide at {}: {}",
                style_guide_path.display(),
                e
            );
            String::new()
        });

        let raw_content = data.get("content").cloned().unwrap_or(Value::Null);
        let content_str = match &raw_content {
            Value::Object(_) | Value::Array(_) => serde_json::to_string(&raw_content)?,
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        let visual_intent = data
            .get("visual_intent")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let user_prompt = format!(
            "Detailed Description: {}\nStyle Guidelines: {}\n{}: {}\n{}: {}\nYour Output:",
            detailed_description,
            style_guide,
            cfg.context_labels[0],
            content_str,
            cfg.context_labels[1],
            visual_intent,
        );

        let contents = vec![ContentItem {
            content_type: "text".into(),
            text: Some(user_prompt),
            source: None,
            image_base64: None,
        }];

        println!(
            "[Stylist] Refining description for {} task...",
            cfg.task_name
        );

        let response_list = call_text_model_with_retry_async(
            clients,
            &self.config.model_name,
            &contents,
            &self.config.system_prompt,
            exp.temperature,
            1,
            50000,
            5,
            5,
            "stylist_agent",
        )
        .await?;

        if let Some(response) = response_list.first() {
            data.insert(output_desc_key, Value::String(response.clone()));
        }

        Ok(data)
    }
}

const DIAGRAM_STYLIST_AGENT_SYSTEM_PROMPT: &str = r#"
## ROLE
You are a Lead Visual Designer for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
Our goal is to generate high-quality, publication-ready diagrams. A planner agent has already generated a preliminary description of the target diagram. Your task is to refine and enrich this description based on the provided [NeurIPS 2025 Style Guidelines] to ensure the final generated image is publication-ready.

**Crucial Instructions:**
1. **Preserve Semantic Content:** Do NOT alter the semantic content, logic, or structure of the diagram.
2. **Preserve High-Quality Aesthetics and Intervene Only When Necessary.**
3. **Respect Diversity:** Different domains have different styles. Keep what works.
4. **Enrich Details:** If the input is plain, enrich it with specific visual attributes.
5. **Handle Icons with Care:** Verify semantic meanings before modifying icons.

## OUTPUT
Output ONLY the final polished Detailed Description. Do not include any conversational text or explanations.
"#;

const PLOT_STYLIST_AGENT_SYSTEM_PROMPT: &str = r#"
## ROLE
You are a Lead Visual Designer for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You are provided with a preliminary description of a statistical plot. Your task is to refine and enrich this description based on the provided [NeurIPS 2025 Style Guidelines] to ensure the final generated image is publication-ready.

**Crucial Instructions:**
1. **Enrich Details:** Focus on specifying visual attributes (colors, fonts, line styles, layout adjustments).
2. **Preserve Content:** Do NOT alter the semantic content, logic, or quantitative results.
3. **Context Awareness:** Use the provided data and visual intent for context.

## OUTPUT
Output ONLY the final polished Detailed Description. Do not include any conversational text or explanations.
"#;
