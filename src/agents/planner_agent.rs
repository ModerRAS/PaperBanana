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

//! Planner Agent - generates detailed figure descriptions from method sections
//! using retrieved reference examples.

use anyhow::Result;
use async_trait::async_trait;
use base64::Engine;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;

use crate::agents::base_agent::{Agent, AgentConfig};
use crate::config::ExpConfig;
use crate::generation_utils::{call_text_model_with_retry_async, ApiClients, ContentItem};

struct TaskConfig {
    task_name: String,
    content_label: &'static str,
    visual_intent_label: &'static str,
}

pub struct PlannerAgent {
    config: AgentConfig,
    task_config: TaskConfig,
}

impl PlannerAgent {
    pub fn new(exp_config: ExpConfig) -> Self {
        let (system_prompt, task_config) = if exp_config.task_name.contains("plot") {
            (
                PLOT_PLANNER_AGENT_SYSTEM_PROMPT.to_string(),
                TaskConfig {
                    task_name: "plot".into(),
                    content_label: "Plot Raw Data",
                    visual_intent_label: "Visual Intent of the Desired Plot",
                },
            )
        } else {
            (
                DIAGRAM_PLANNER_AGENT_SYSTEM_PROMPT.to_string(),
                TaskConfig {
                    task_name: "diagram".into(),
                    content_label: "Methodology Section",
                    visual_intent_label: "Diagram Caption",
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
impl Agent for PlannerAgent {
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
        let description = data
            .get("visual_intent")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let mut content_list: Vec<ContentItem> = Vec::new();

        // Load examples: prefer pre-loaded retrieved_examples, otherwise load from ref.json
        let examples: Vec<Value> = {
            let pre_loaded = data
                .get("retrieved_examples")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            if !pre_loaded.is_empty() {
                pre_loaded
            } else {
                let retrieved_ids: Vec<String> = data
                    .get("top10_references")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();

                if retrieved_ids.is_empty() {
                    vec![]
                } else {
                    let ref_path = exp
                        .work_dir
                        .join(format!("data/PaperBananaBench/{}/ref.json", cfg.task_name));
                    let raw = fs::read_to_string(&ref_path).unwrap_or_else(|_| "[]".into());
                    let pool: Vec<Value> = serde_json::from_str(&raw).unwrap_or_default();
                    let id_map: HashMap<String, &Value> = pool
                        .iter()
                        .filter_map(|v| {
                            v.get("id")
                                .and_then(|id| id.as_str())
                                .map(|id| (id.to_string(), v))
                        })
                        .collect();
                    retrieved_ids
                        .iter()
                        .filter_map(|id| id_map.get(id).map(|v| (*v).clone()))
                        .collect()
                }
            }
        };

        let mut user_prompt = String::new();
        for (idx, item) in examples.iter().enumerate() {
            user_prompt.push_str(&format!("Example {}:\n", idx + 1));

            let item_content = item.get("content").cloned().unwrap_or(Value::Null);
            let item_content_str = match &item_content {
                Value::Object(_) | Value::Array(_) => serde_json::to_string(&item_content)?,
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };

            user_prompt.push_str(&format!("{}: {}\n", cfg.content_label, item_content_str));
            user_prompt.push_str(&format!(
                "{}: {}\nReference {}: ",
                cfg.visual_intent_label,
                item.get("visual_intent")
                    .and_then(|v| v.as_str())
                    .unwrap_or(""),
                capitalize(&cfg.task_name),
            ));

            content_list.push(ContentItem {
                content_type: "text".into(),
                text: Some(user_prompt.clone()),
                source: None,
                image_base64: None,
            });
            user_prompt.clear();

            // Load reference image
            if let Some(rel_path) = item.get("path_to_gt_image").and_then(|v| v.as_str()) {
                let image_path = exp.work_dir.join(format!(
                    "data/PaperBananaBench/{}/{}",
                    cfg.task_name, rel_path
                ));
                if let Ok(bytes) = fs::read(&image_path) {
                    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
                    content_list.push(ContentItem {
                        content_type: "image".into(),
                        text: None,
                        source: None,
                        image_base64: Some(b64),
                    });
                }
            }
        }

        user_prompt.push_str(&format!(
            "Now, based on the following {} and {}, provide a detailed description for the figure to be generated.\n",
            cfg.content_label.to_lowercase(),
            cfg.visual_intent_label.to_lowercase(),
        ));
        user_prompt.push_str(&format!(
            "{}: {}\n{}: {}\n",
            cfg.content_label, content, cfg.visual_intent_label, description,
        ));
        user_prompt.push_str("Detailed description of the target figure to be generated");
        if cfg.task_name == "diagram" {
            user_prompt.push_str(" (do not include figure titles)");
        }
        user_prompt.push(':');

        content_list.push(ContentItem {
            content_type: "text".into(),
            text: Some(user_prompt),
            source: None,
            image_base64: None,
        });

        println!(
            "[Planner] Generating description for {} task...",
            cfg.task_name
        );

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
            "planner_agent",
        )
        .await?;

        for (idx, response) in response_list.iter().enumerate() {
            data.insert(
                format!("target_{}_desc{}", cfg.task_name, idx),
                Value::String(response.trim().to_string()),
            );
        }

        Ok(data)
    }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().to_string() + c.as_str(),
    }
}

const DIAGRAM_PLANNER_AGENT_SYSTEM_PROMPT: &str = r#"
I am working on a task: given the 'Methodology' section of a paper, and the caption of the desired figure, automatically generate a corresponding illustrative diagram. I will input the text of the 'Methodology' section, the figure caption, and your output should be a detailed description of an illustrative figure that effectively represents the methods described in the text.

To help you understand the task better, and grasp the principles for generating such figures, I will also provide you with several examples. You should learn from these examples to provide your figure description.

** IMPORTANT: **
Your description should be as detailed as possible. Semantically, clearly describe each element and their connections. Formally, include various details such as background style (typically pure white or very light pastel), colors, line thickness, icon styles, etc. Remember: vague or unclear specifications will only make the generated figure worse, not better.
"#;

const PLOT_PLANNER_AGENT_SYSTEM_PROMPT: &str = r#"
I am working on a task: given the raw data (typically in tabular or json format) and a visual intent of the desired plot, automatically generate a corresponding statistical plot that are both accurate and aesthetically pleasing. I will input the raw data and the plot visual intent, and your output should be a detailed description of an illustrative plot that effectively represents the data. Note that your description should include all the raw data points to be plotted.

To help you understand the task better, and grasp the principles for generating such plots, I will also provide you with several examples. You should learn from these examples to provide your plot description.

** IMPORTANT: **
Your description should be as detailed as possible. For content, explain the precise mapping of variables to visual channels (x, y, hue) and explicitly enumerate every raw data point's coordinate to be drawn to ensure accuracy. For presentation, specify the exact aesthetic parameters, including specific HEX color codes, font sizes for all labels, line widths, marker dimensions, legend placement, and grid styles. You should learn from the examples' content presentation and aesthetic design (e.g., color schemes).
"#;
