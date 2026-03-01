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

//! Polish Agent - polishes existing images by generating improvement suggestions
//! and then regenerating with those suggestions applied.

use anyhow::Result;
use async_trait::async_trait;
use base64::Engine;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;

use crate::agents::base_agent::{Agent, AgentConfig};
use crate::config::ExpConfig;
use crate::generation_utils::{
    call_gemini_with_retry_async, call_text_model_with_retry_async, ApiClients, ContentItem,
    ImageSource,
};
use crate::image_utils::convert_png_b64_to_jpg_b64;

pub struct PolishAgent {
    config: AgentConfig,
    image_model_name: String,
    text_model_name: String,
    style_guide_filename: String,
    suggestion_system_prompt: String,
    task_name: String,
}

impl PolishAgent {
    pub fn new(exp_config: ExpConfig) -> Self {
        let (system_prompt, suggestion_prompt, style_file, task_name) =
            if exp_config.task_name == "plot" {
                (
                    PLOT_POLISH_AGENT_SYSTEM_PROMPT.to_string(),
                    PLOT_SUGGESTION_SYSTEM_PROMPT.to_string(),
                    "neurips2025_plot_style_guide.md".to_string(),
                    "plot".to_string(),
                )
            } else {
                (
                    DIAGRAM_POLISH_AGENT_SYSTEM_PROMPT.to_string(),
                    DIAGRAM_SUGGESTION_SYSTEM_PROMPT.to_string(),
                    "neurips2025_diagram_style_guide.md".to_string(),
                    "diagram".to_string(),
                )
            };

        let image_model_name = exp_config.image_model_name.clone();
        let text_model_name = exp_config.model_name.clone();

        Self {
            config: AgentConfig {
                model_name: image_model_name.clone(),
                system_prompt,
                exp_config,
            },
            image_model_name,
            text_model_name,
            style_guide_filename: style_file,
            suggestion_system_prompt: suggestion_prompt,
            task_name,
        }
    }

    async fn generate_suggestions(
        &self,
        gt_image_b64: &str,
        style_guide: &str,
        clients: &ApiClients,
    ) -> Result<String> {
        let user_prompt = format!(
            "Here is the style guide:\n{}\n\nPlease analyze the provided image against this style guide and list up to 10 specific improvement suggestions to make the image visually more appealing. If the image is already perfect, just say 'No changes needed'.",
            style_guide
        );

        let contents = vec![
            ContentItem {
                content_type: "text".into(),
                text: Some(user_prompt),
                source: None,
                image_base64: None,
            },
            ContentItem {
                content_type: "image".into(),
                text: None,
                source: Some(ImageSource {
                    source_type: "base64".into(),
                    media_type: Some("image/jpeg".into()),
                    data: Some(gt_image_b64.to_string()),
                }),
                image_base64: None,
            },
        ];

        let response_list = call_text_model_with_retry_async(
            clients,
            &self.text_model_name,
            &contents,
            &self.suggestion_system_prompt,
            1.0,
            1,
            50000,
            3,
            10,
            "polish_agent_suggestions",
        )
        .await?;

        Ok(response_list.into_iter().next().unwrap_or_default())
    }
}

#[async_trait]
impl Agent for PolishAgent {
    async fn process(
        &self,
        mut data: HashMap<String, Value>,
        clients: &ApiClients,
    ) -> Result<HashMap<String, Value>> {
        let exp = &self.config.exp_config;

        let gt_image_path_rel = match data.get("path_to_gt_image").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => {
                println!("[Polish] No GT image path found in data");
                return Ok(data);
            }
        };

        let gt_image_path = exp.work_dir.join(format!(
            "data/PaperBananaBench/{}/{}",
            self.task_name, gt_image_path_rel
        ));

        let gt_image_b64 = match fs::read(&gt_image_path) {
            Ok(bytes) => base64::engine::general_purpose::STANDARD.encode(&bytes),
            Err(e) => {
                println!(
                    "[Polish] Failed to load GT image from {}: {}",
                    gt_image_path.display(),
                    e
                );
                return Ok(data);
            }
        };

        let style_guide_path = exp
            .work_dir
            .join("style_guides")
            .join(&self.style_guide_filename);
        let style_guide = match fs::read_to_string(&style_guide_path) {
            Ok(s) => s,
            Err(e) => {
                println!(
                    "[Polish] Error loading style guide from {}: {}",
                    style_guide_path.display(),
                    e
                );
                return Ok(data);
            }
        };

        // Step 1: Generate suggestions
        println!(
            "[Polish] Step 1: Generating suggestions for {} ...",
            self.task_name
        );
        let suggestions = self
            .generate_suggestions(&gt_image_b64, &style_guide, clients)
            .await?;

        if suggestions.contains("No changes needed") {
            println!("[Polish] No changes needed for this image.");
        }

        if !suggestions.is_empty() {
            data.insert(
                format!("suggestions_{}", self.task_name),
                Value::String(suggestions.clone()),
            );
        }
        println!(
            "[Polish] Suggestions: {}...",
            &suggestions[..suggestions.len().min(200)]
        );

        // Step 2: Polish image using suggestions
        println!("[Polish] Step 2: Polishing image with suggestions...");
        let user_prompt = format!(
            "Please polish this image based on the following suggestions:\n\n{}\n\nPolished Image:",
            suggestions
        );

        let contents = vec![
            ContentItem {
                content_type: "text".into(),
                text: Some(user_prompt),
                source: None,
                image_base64: None,
            },
            ContentItem {
                content_type: "image".into(),
                text: None,
                source: Some(ImageSource {
                    source_type: "base64".into(),
                    media_type: Some("image/jpeg".into()),
                    data: Some(gt_image_b64),
                }),
                image_base64: None,
            },
        ];

        let aspect_ratio = data
            .get("additional_info")
            .and_then(|v| v.get("rounded_ratio"))
            .and_then(|v| v.as_str())
            .unwrap_or("16:9")
            .to_string();

        let response_list = call_gemini_with_retry_async(
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
            &format!("polish_agent_{}", self.task_name),
        )
        .await;

        // Store aspect_ratio for potential downstream use
        let _ = aspect_ratio;

        if let Some(first) = response_list.first() {
            if !first.is_empty() {
                if let Some(jpg) = convert_png_b64_to_jpg_b64(Some(first)) {
                    let output_key = format!("polished_{}_base64_jpg", self.task_name);
                    data.insert(output_key, Value::String(jpg));
                } else {
                    println!("[Polish] Image conversion failed");
                }
            } else {
                println!("[Polish] No response from model");
            }
        } else {
            println!("[Polish] No response from model");
        }

        Ok(data)
    }
}

const DIAGRAM_SUGGESTION_SYSTEM_PROMPT: &str = r#"
You are a senior art director for NeurIPS 2025. Your task is to critique a diagram against a provided style guide.
Provide up to 10 concise, actionable improvement suggestions. Focus on aesthetics (color, layout, fonts, icons).
Directly list the suggestions. Do not use filler phrases like "Based on the style guide...".
If the diagram is substantially compliant, output "No changes needed".
"#;

const PLOT_SUGGESTION_SYSTEM_PROMPT: &str = r#"
You are a senior data visualization expert for NeurIPS 2025. Your task is to critique a plot against a provided style guide.
Provide up to 10 concise, actionable improvement suggestions. Focus on aesthetics (color, layout, fonts).
Directly list the suggestions. Do not use filler phrases like "Based on the style guide...".
If the plot is substantially compliant, output "No changes needed".
"#;

const DIAGRAM_POLISH_AGENT_SYSTEM_PROMPT: &str = r#"
## ROLE
You are a professional diagram polishing expert for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You are given an existing diagram image and a list of specific improvement suggestions. Your task is to generate a polished version of this diagram by applying these suggestions while preserving the semantic logic and structure of the original diagram.

## OUTPUT
Generate a polished diagram image that maintains the original content while applying the improvement suggestions.
"#;

const PLOT_POLISH_AGENT_SYSTEM_PROMPT: &str = r#"
## ROLE
You are a professional plot polishing expert for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You are given an existing statistical plot image and a list of specific improvement suggestions. Your task is to generate a polished version of this plot by applying these suggestions while preserving all the data and quantitative information.

**Important Instructions:**
1. **Preserve Data:** Do NOT alter any data points, values, or quantitative information.
2. **Apply Suggestions:** Enhance the visual aesthetics according to the provided suggestions.
3. **Maintain Accuracy:** Ensure all numerical values and relationships remain accurate.
4. **Professional Quality:** Ensure the output meets publication standards.

## OUTPUT
Generate a polished plot image that maintains the original data while applying the improvement suggestions.
"#;
