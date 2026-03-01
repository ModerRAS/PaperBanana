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

//! Critic Agent - critiques and refines figure descriptions.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

use crate::agents::base_agent::{Agent, AgentConfig};
use crate::config::ExpConfig;
use crate::generation_utils::{call_text_model_with_retry_async, ApiClients, ContentItem, ImageSource};

struct TaskConfig {
    task_name: String,
    critique_target: &'static str,
    context_labels: [&'static str; 2],
}

pub struct CriticAgent {
    config: AgentConfig,
    task_config: TaskConfig,
}

impl CriticAgent {
    pub fn new(exp_config: ExpConfig) -> Self {
        let (system_prompt, task_config) = if exp_config.task_name == "plot" {
            (
                PLOT_CRITIC_AGENT_SYSTEM_PROMPT.to_string(),
                TaskConfig {
                    task_name: "plot".into(),
                    critique_target: "Target Plot for Critique:",
                    context_labels: ["Raw Data", "Visual Intent"],
                },
            )
        } else {
            (
                DIAGRAM_CRITIC_AGENT_SYSTEM_PROMPT.to_string(),
                TaskConfig {
                    task_name: "diagram".into(),
                    critique_target: "Target Diagram for Critique:",
                    context_labels: ["Methodology Section", "Figure Caption"],
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

    /// Process with explicit source and round control.
    pub async fn process_with_options(
        &self,
        mut data: HashMap<String, Value>,
        clients: &ApiClients,
        source: &str,
    ) -> Result<HashMap<String, Value>> {
        let cfg = &self.task_config;
        let exp = &self.config.exp_config;
        let task_name = &cfg.task_name;

        let round_idx = data
            .get("current_critic_round")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let (desc_key, base64_key) = if round_idx == 0 {
            match source {
                "stylist" => (
                    format!("target_{}_stylist_desc0", task_name),
                    format!("target_{}_stylist_desc0_base64_jpg", task_name),
                ),
                "planner" => (
                    format!("target_{}_desc0", task_name),
                    format!("target_{}_desc0_base64_jpg", task_name),
                ),
                _ => anyhow::bail!("Invalid source '{}'. Must be 'stylist' or 'planner'.", source),
            }
        } else {
            (
                format!("target_{}_critic_desc{}", task_name, round_idx - 1),
                format!(
                    "target_{}_critic_desc{}_base64_jpg",
                    task_name,
                    round_idx - 1
                ),
            )
        };

        let detailed_description = data
            .get(&desc_key)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let image_base64 = data
            .get(&base64_key)
            .and_then(|v| v.as_str())
            .map(String::from);

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

        let mut content_list: Vec<ContentItem> = vec![ContentItem {
            content_type: "text".into(),
            text: Some(cfg.critique_target.to_string()),
            source: None,
            image_base64: None,
        }];

        // Add image or text-only notice
        if let Some(ref b64) = image_base64 {
            if b64.len() > 100 {
                content_list.push(ContentItem {
                    content_type: "image".into(),
                    text: None,
                    source: Some(ImageSource {
                        source_type: "base64".into(),
                        media_type: Some("image/jpeg".into()),
                        data: Some(b64.clone()),
                    }),
                    image_base64: None,
                });
            } else {
                println!(
                    "[Critic] No valid image found for round {}. Using text-only critique mode.",
                    round_idx
                );
                content_list.push(ContentItem {
                    content_type: "text".into(),
                    text: Some("\n[SYSTEM NOTICE] The plot image could not be generated based on the current description (likely due to invalid code). Please check the description for errors and provide a revised version.".into()),
                    source: None,
                    image_base64: None,
                });
            }
        } else {
            println!(
                "[Critic] No image available for round {}. Using text-only critique mode.",
                round_idx
            );
            content_list.push(ContentItem {
                content_type: "text".into(),
                text: Some("\n[SYSTEM NOTICE] The image could not be generated. Please check the description for errors and provide a revised version.".into()),
                source: None,
                image_base64: None,
            });
        }

        content_list.push(ContentItem {
            content_type: "text".into(),
            text: Some(format!(
                "Detailed Description: {}\n{}: {}\n{}: {}\nYour Output:",
                detailed_description, cfg.context_labels[0], content_str, cfg.context_labels[1], visual_intent,
            )),
            source: None,
            image_base64: None,
        });

        println!("[Critic] Round {} critique for {} task...", round_idx, task_name);

        let response_list = call_text_model_with_retry_async(
            clients,
            &self.config.model_name,
            &content_list,
            &self.config.system_prompt,
            exp.temperature,
            1,
            50000,
            5,
            5,
            "critic_agent",
        )
        .await?;

        let cleaned = response_list
            .first()
            .map(|r| r.replace("```json", "").replace("```", "").trim().to_string())
            .unwrap_or_default();

        let eval_result: Value = serde_json::from_str(&cleaned).unwrap_or_else(|e| {
            println!("[Critic] JSON parse error: {}. Raw: {}...", e, &cleaned[..cleaned.len().min(200)]);
            Value::Object(serde_json::Map::new())
        });

        let critic_suggestions = eval_result
            .get("critic_suggestions")
            .and_then(|v| v.as_str())
            .unwrap_or("No changes needed.")
            .to_string();
        let revised_description = eval_result
            .get("revised_description")
            .and_then(|v| v.as_str())
            .unwrap_or("No changes needed.")
            .to_string();

        data.insert(
            format!("target_{}_critic_suggestions{}", task_name, round_idx),
            Value::String(critic_suggestions),
        );

        if revised_description.trim() == "No changes needed." {
            data.insert(
                format!("target_{}_critic_desc{}", task_name, round_idx),
                Value::String(detailed_description),
            );
        } else {
            data.insert(
                format!("target_{}_critic_desc{}", task_name, round_idx),
                Value::String(revised_description),
            );
        }

        Ok(data)
    }
}

#[async_trait]
impl Agent for CriticAgent {
    async fn process(
        &self,
        data: HashMap<String, Value>,
        clients: &ApiClients,
    ) -> Result<HashMap<String, Value>> {
        self.process_with_options(data, clients, "stylist").await
    }
}

const DIAGRAM_CRITIC_AGENT_SYSTEM_PROMPT: &str = r#"
## ROLE
You are a Lead Visual Designer for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
Conduct a sanity check and provide a critique of the target diagram based on its content and presentation. Ensure alignment with the provided 'Methodology Section' and 'Figure Caption'.

## CRITIQUE & REVISION RULES

1. Content
   - **Fidelity & Alignment:** Ensure the diagram accurately reflects the methodology. No hallucinated content.
   - **Text QA:** Check for typos, nonsensical text, or unclear labels.
   - **Validation of Examples:** Verify illustrative examples are factually correct.
   - **Caption Exclusion:** The figure caption must NOT appear inside the image.

2. Presentation
   - **Clarity & Readability:** Evaluate overall visual clarity.
   - **Legend Management:** Remove redundant text-based legends.

Your description should primarily modify the original, not rewrite from scratch.

## OUTPUT
```json
{
    "critic_suggestions": "Detailed critique or 'No changes needed.'",
    "revised_description": "Fully revised description or 'No changes needed.'"
}
```
"#;

const PLOT_CRITIC_AGENT_SYSTEM_PROMPT: &str = r#"
## ROLE
You are a Lead Visual Designer for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
Conduct a sanity check and provide a critique of the target plot. Ensure alignment with the provided 'Raw Data' and 'Visual Intent'.

## CRITIQUE & REVISION RULES

1. Content
   - **Data Fidelity & Alignment:** All quantitative values must be correct. No data hallucinated or omitted.
   - **Text QA:** Check for typos and unclear labels.
   - **Validation of Values:** Verify numerical values, axis scales, and data points.
   - **Caption Exclusion:** The figure caption must NOT appear inside the image.

2. Presentation
   - **Clarity & Readability:** Evaluate visual clarity.
   - **Overlap & Layout:** Check for overlapping elements that reduce readability.
   - **Legend Management:** Remove redundant text-based legends.

3. Handling Generation Failures
   - If the target plot is missing or replaced by a system notice, analyze the description for errors and provide a simplified, robust revision.

## OUTPUT
```json
{
    "critic_suggestions": "Detailed critique or 'No changes needed.'",
    "revised_description": "Fully revised description or 'No changes needed.'"
}
```
"#;
