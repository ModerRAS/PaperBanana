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

//! Vanilla Agent - directly generates images from method sections or code for plots.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

use crate::agents::base_agent::{Agent, AgentConfig};
use crate::config::ExpConfig;
use crate::generation_utils::{
    call_doubao_image_generation_with_retry_async, call_gemini_with_retry_async,
    call_openai_image_generation_with_retry_async, call_text_model_with_retry_async, ApiClients,
    ContentItem, ImageGenConfig,
};
use crate::image_utils::convert_png_b64_to_jpg_b64;

struct TaskConfig {
    task_name: String,
    use_image_generation: bool,
    content_label: &'static str,
    visual_intent_label: &'static str,
}

pub struct VanillaAgent {
    config: AgentConfig,
    task_config: TaskConfig,
    image_model_name: String,
}

impl VanillaAgent {
    pub fn new(exp_config: ExpConfig) -> Self {
        let (system_prompt, task_config, image_model_name) =
            if exp_config.task_name.contains("plot") {
                (
                    PLOT_VANILLA_AGENT_SYSTEM_PROMPT.to_string(),
                    TaskConfig {
                        task_name: "plot".into(),
                        use_image_generation: false,
                        content_label: "Plot Raw Data",
                        visual_intent_label: "Visual Intent of the Desired Plot",
                    },
                    exp_config.model_name.clone(),
                )
            } else {
                (
                    DIAGRAM_VANILLA_AGENT_SYSTEM_PROMPT.to_string(),
                    TaskConfig {
                        task_name: "diagram".into(),
                        use_image_generation: true,
                        content_label: "Method Section",
                        visual_intent_label: "Diagram Caption",
                    },
                    exp_config.image_model_name.clone(),
                )
            };

        let model_name = if task_config.use_image_generation {
            exp_config.image_model_name.clone()
        } else {
            exp_config.model_name.clone()
        };

        Self {
            config: AgentConfig {
                model_name,
                system_prompt,
                exp_config,
            },
            task_config,
            image_model_name,
        }
    }
}

#[async_trait]
impl Agent for VanillaAgent {
    async fn process(
        &self,
        mut data: HashMap<String, Value>,
        clients: &ApiClients,
    ) -> Result<HashMap<String, Value>> {
        let cfg = &self.task_config;
        let exp = &self.config.exp_config;

        let raw_content = data.get("content").cloned().unwrap_or(Value::Null);
        let content = match &raw_content {
            Value::Object(_) | Value::Array(_) => serde_json::to_string(&raw_content)?,
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        let visual_intent = data
            .get("visual_intent")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let mut prompt_text = format!(
            "**{}**: {}\n**{}**: {}\n",
            cfg.content_label, content, cfg.visual_intent_label, visual_intent,
        );
        if cfg.task_name == "diagram" {
            prompt_text.push_str("Note that do not include figure titles in the image.");
        }

        if cfg.use_image_generation {
            prompt_text.push_str("**Generated Diagram**: ");
        } else {
            prompt_text.push_str("\nUse python matplotlib to generate a statistical plot based on the above information. Only provide the code without any explanations. Code:");
        }

        println!("[Vanilla] Generating {} ...", cfg.task_name);

        let output_key = format!("vanilla_{}_base64_jpg", cfg.task_name);

        if cfg.use_image_generation {
            let response_list = if self.image_model_name.contains("gemini") {
                let contents = vec![ContentItem {
                    content_type: "text".into(),
                    text: Some(prompt_text.clone()),
                    source: None,
                    image_base64: None,
                }];
                call_gemini_with_retry_async(
                    clients,
                    &self.image_model_name,
                    &contents,
                    &self.config.system_prompt,
                    exp.temperature,
                    1,
                    50000,
                    5,
                    30,
                    Some(vec!["IMAGE".into()]),
                    "vanilla_agent_gemini",
                )
                .await
            } else if self.image_model_name.contains("gpt-image") {
                let truncated = if prompt_text.len() > 30000 {
                    &prompt_text[..30000]
                } else {
                    &prompt_text
                };
                let config = ImageGenConfig {
                    size: "1536x1024".into(),
                    quality: "high".into(),
                    background: "opaque".into(),
                    output_format: "png".into(),
                    ..Default::default()
                };
                call_openai_image_generation_with_retry_async(
                    clients,
                    &self.image_model_name,
                    truncated,
                    &config,
                    5,
                    30,
                    "vanilla_agent_openai",
                )
                .await
            } else if self.image_model_name.contains("doubao") {
                let truncated = if prompt_text.len() > 30000 {
                    &prompt_text[..30000]
                } else {
                    &prompt_text
                };
                let config = ImageGenConfig {
                    size: "1024x1024".into(),
                    response_format: "b64_json".into(),
                    ..Default::default()
                };
                call_doubao_image_generation_with_retry_async(
                    clients,
                    &self.image_model_name,
                    truncated,
                    &config,
                    5,
                    30,
                    "vanilla_agent_doubao",
                )
                .await
            } else {
                anyhow::bail!("Unsupported image model: {}", self.image_model_name);
            };

            if let Some(first) = response_list.first() {
                if let Some(jpg) = convert_png_b64_to_jpg_b64(Some(first)) {
                    data.insert(output_key, Value::String(jpg));
                }
            }
        } else {
            // Text model for code generation (plots)
            let contents = vec![ContentItem {
                content_type: "text".into(),
                text: Some(prompt_text),
                source: None,
                image_base64: None,
            }];
            let response_list = call_text_model_with_retry_async(
                clients,
                &self.config.model_name,
                &contents,
                &self.config.system_prompt,
                exp.temperature,
                1,
                50000,
                5,
                30,
                "vanilla_agent_text",
            )
            .await?;

            if let Some(raw_code) = response_list.first() {
                if !raw_code.is_empty() {
                    // Store code; actual execution requires Python runtime
                    data.insert(
                        format!("vanilla_{}_code", cfg.task_name),
                        Value::String(raw_code.clone()),
                    );
                    println!(
                        "[Vanilla] Generated plot code (execution requires Python runtime)"
                    );
                }
            }
        }

        Ok(data)
    }
}

const DIAGRAM_VANILLA_AGENT_SYSTEM_PROMPT: &str = r#"
## ROLE
You are a Lead Visual Designer for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You will be provided with a "Method Section" and a "Diagram Caption". Your task is to generate a high-quality scientific diagram that effectively illustrates the method described in the text.

**CRITICAL INSTRUCTION ON CAPTION:**
The "Diagram Caption" is provided solely to describe the visual content and logic you need to draw. **DO NOT render, write, or include the caption text itself (e.g., "Figure 1: ...") inside the generated image.**

## OUTPUT
Generate a single, high-resolution image that visually explains the method and aligns well with the caption.
"#;

const PLOT_VANILLA_AGENT_SYSTEM_PROMPT: &str = r#"
## ROLE
You are an expert statistical plot illustrator for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You will be provided with "Plot Raw Data" and a "Visual Intent of the Desired Plot". Your task is to write matplotlib code to generate a high-quality statistical plot that effectively visualizes the data.

## OUTPUT
Write Python matplotlib code to generate the plot. Only provide the code without any explanations.
"#;
