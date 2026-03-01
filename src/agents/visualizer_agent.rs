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

//! Visualizer Agent - generates images from descriptions.
//! For diagrams: uses image generation models directly.
//! For plots: generates matplotlib code (text model).

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
    prompt_template: String,
    max_output_tokens: usize,
}

pub struct VisualizerAgent {
    config: AgentConfig,
    task_config: TaskConfig,
    image_model_name: String,
}

impl VisualizerAgent {
    pub fn new(exp_config: ExpConfig) -> Self {
        let (system_prompt, task_config, image_model_name) = if exp_config
            .task_name
            .contains("plot")
        {
            (
                    PLOT_VISUALIZER_AGENT_SYSTEM_PROMPT.to_string(),
                    TaskConfig {
                        task_name: "plot".into(),
                        use_image_generation: false,
                        prompt_template: "Use python matplotlib to generate a statistical plot based on the following detailed description: {desc}\n Only provide the code without any explanations. Code:".into(),
                        max_output_tokens: 50000,
                    },
                    exp_config.model_name.clone(),
                )
        } else {
            (
                    DIAGRAM_VISUALIZER_AGENT_SYSTEM_PROMPT.to_string(),
                    TaskConfig {
                        task_name: "diagram".into(),
                        use_image_generation: true,
                        prompt_template: "Render an image based on the following detailed description: {desc}\n Note that do not include figure titles in the image. Diagram: ".into(),
                        max_output_tokens: 50000,
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
impl Agent for VisualizerAgent {
    async fn process(
        &self,
        mut data: HashMap<String, Value>,
        clients: &ApiClients,
    ) -> Result<HashMap<String, Value>> {
        let cfg = &self.task_config;
        let exp = &self.config.exp_config;
        let task_name = &cfg.task_name;

        // Collect description keys to process
        let mut desc_keys: Vec<String> = Vec::new();
        for key in &[
            format!("target_{}_desc0", task_name),
            format!("target_{}_stylist_desc0", task_name),
        ] {
            let base64_key = format!("{}_base64_jpg", key);
            if data.contains_key(key) && !data.contains_key(&base64_key) {
                desc_keys.push(key.clone());
            }
        }

        for round_idx in 0..3 {
            let key = format!("target_{}_critic_desc{}", task_name, round_idx);
            let base64_key = format!("{}_base64_jpg", key);
            if data.contains_key(&key) && !data.contains_key(&base64_key) {
                let suggestions_key =
                    format!("target_{}_critic_suggestions{}", task_name, round_idx);
                let suggestions = data
                    .get(&suggestions_key)
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                if suggestions.trim() == "No changes needed." && round_idx > 0 {
                    let prev_key = format!(
                        "target_{}_critic_desc{}_base64_jpg",
                        task_name,
                        round_idx - 1
                    );
                    if let Some(prev_b64) = data.get(&prev_key).cloned() {
                        data.insert(base64_key, prev_b64);
                        println!(
                            "[Visualizer] Reused base64 from round {} for {}",
                            round_idx - 1,
                            key
                        );
                        continue;
                    }
                }
                desc_keys.push(key);
            }
        }

        for desc_key in &desc_keys {
            let desc_value = data.get(desc_key).and_then(|v| v.as_str()).unwrap_or("");
            let prompt_text = cfg.prompt_template.replace("{desc}", desc_value);

            println!("[Visualizer] Processing {} ...", desc_key);

            let response_list: Vec<String> = if cfg.use_image_generation {
                if self.image_model_name.contains("gemini") {
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
                        cfg.max_output_tokens,
                        5,
                        30,
                        Some(vec!["IMAGE".into()]),
                        "visualizer_agent_gemini",
                    )
                    .await
                } else if self.image_model_name.contains("gpt-image") {
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
                        &prompt_text,
                        &config,
                        5,
                        30,
                        "visualizer_agent_openai",
                    )
                    .await
                } else if self.image_model_name.contains("doubao") {
                    let config = ImageGenConfig {
                        size: "1024x1024".into(),
                        response_format: "b64_json".into(),
                        ..Default::default()
                    };
                    call_doubao_image_generation_with_retry_async(
                        clients,
                        &self.image_model_name,
                        &prompt_text,
                        &config,
                        5,
                        30,
                        "visualizer_agent_doubao",
                    )
                    .await
                } else {
                    println!(
                        "[Visualizer] Unsupported image model: {}",
                        self.image_model_name
                    );
                    continue;
                }
            } else {
                // Text model for code generation (plots)
                let contents = vec![ContentItem {
                    content_type: "text".into(),
                    text: Some(prompt_text),
                    source: None,
                    image_base64: None,
                }];
                call_text_model_with_retry_async(
                    clients,
                    &self.config.model_name,
                    &contents,
                    &self.config.system_prompt,
                    exp.temperature,
                    1,
                    cfg.max_output_tokens,
                    5,
                    30,
                    "visualizer_agent_text",
                )
                .await
                .unwrap_or_default()
            };

            if response_list.is_empty() || response_list[0].is_empty() {
                println!("[Visualizer] No response for {}", desc_key);
                continue;
            }

            let output_key = format!("{}_base64_jpg", desc_key);
            if cfg.use_image_generation {
                if let Some(jpg) = convert_png_b64_to_jpg_b64(Some(&response_list[0])) {
                    data.insert(output_key, Value::String(jpg));
                } else {
                    println!("[Visualizer] Image conversion failed for {}", desc_key);
                }
            } else {
                // For plots, store the generated code
                data.insert(
                    format!("{}_code", desc_key),
                    Value::String(response_list[0].clone()),
                );
                // Note: actual matplotlib code execution would require a Python subprocess.
                // Store the code for downstream execution.
                println!(
                    "[Visualizer] Generated plot code for {} (execution requires Python runtime)",
                    desc_key
                );
            }
        }

        Ok(data)
    }
}

const DIAGRAM_VISUALIZER_AGENT_SYSTEM_PROMPT: &str =
    "You are an expert scientific diagram illustrator. Generate high-quality scientific diagrams based on user requests.";

const PLOT_VISUALIZER_AGENT_SYSTEM_PROMPT: &str =
    "You are an expert statistical plot illustrator. Write code to generate high-quality statistical plots based on user requests.";
