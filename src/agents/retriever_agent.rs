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

//! Retriever Agent - retrieves relevant reference examples from the candidate pool.

use anyhow::Result;
use async_trait::async_trait;
use rand::seq::SliceRandom;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;

use crate::agents::base_agent::{Agent, AgentConfig};
use crate::config::ExpConfig;
use crate::generation_utils::{ApiClients, ContentItem};

/// Task-specific configuration for the retriever.
struct TaskConfig {
    task_name: String,
    ref_limit: Option<usize>,
    target_labels: [&'static str; 2],
    candidate_labels: [&'static str; 3],
    candidate_type: &'static str,
    output_key: &'static str,
    instruction_suffix: &'static str,
}

pub struct RetrieverAgent {
    config: AgentConfig,
    task_config: TaskConfig,
}

impl RetrieverAgent {
    pub fn new(exp_config: ExpConfig) -> Self {
        let (system_prompt, task_config) = if exp_config.task_name == "plot" {
            (
                PLOT_RETRIEVER_AGENT_SYSTEM_PROMPT.to_string(),
                TaskConfig {
                    task_name: "plot".into(),
                    ref_limit: None,
                    target_labels: ["Visual Intent", "Raw Data"],
                    candidate_labels: ["Plot ID", "Visual Intent", "Raw Data"],
                    candidate_type: "Plot",
                    output_key: "top10_references",
                    instruction_suffix: "select the Top 10 most relevant plots according to the instructions provided. Your output should be a strictly valid JSON object containing a single list of the exact ids of the top 10 selected plots.",
                },
            )
        } else {
            (
                DIAGRAM_RETRIEVER_AGENT_SYSTEM_PROMPT.to_string(),
                TaskConfig {
                    task_name: "diagram".into(),
                    ref_limit: Some(200),
                    target_labels: ["Caption", "Methodology section"],
                    candidate_labels: ["Diagram ID", "Caption", "Methodology section"],
                    candidate_type: "Diagram",
                    output_key: "top10_references",
                    instruction_suffix: "select the Top 10 most relevant diagrams according to the instructions provided. Your output should be a strictly valid JSON object containing a single list of the exact ids of the top 10 selected diagrams.",
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

    /// Full process method with explicit retrieval_setting parameter.
    pub async fn process_with_setting(
        &self,
        mut data: HashMap<String, Value>,
        clients: &ApiClients,
        retrieval_setting: &str,
    ) -> Result<HashMap<String, Value>> {
        let cfg = &self.task_config;
        let exp = &self.config.exp_config;

        let ref_file = exp
            .work_dir
            .join(format!("data/PaperBananaBench/{}/ref.json", cfg.task_name));

        let mut setting = retrieval_setting.to_string();

        if (setting == "auto" || setting == "random") && !ref_file.exists() {
            println!(
                "Warning: Reference file not found at {}. Falling back to retrieval_setting='none'.",
                ref_file.display()
            );
            setting = "none".into();
        }

        if setting == "manual" {
            let manual_file = exp.work_dir.join(format!(
                "data/PaperBananaBench/{}/agent_selected_12.json",
                cfg.task_name
            ));
            if !manual_file.exists() {
                println!(
                    "Warning: Manual reference file not found at {}. Falling back to 'none'.",
                    manual_file.display()
                );
                setting = "none".into();
            }
        }

        match setting.as_str() {
            "none" => {
                data.insert("top10_references".into(), json!([]));
                data.insert("retrieved_examples".into(), json!([]));
            }
            "manual" => {
                let (ids, examples) = self.load_manual_references()?;
                data.insert("top10_references".into(), json!(ids));
                data.insert("retrieved_examples".into(), examples);
            }
            "random" => {
                let ids = self.load_random_references()?;
                data.insert("top10_references".into(), json!(ids));
                data.insert("retrieved_examples".into(), json!([]));
            }
            "auto" => {
                let ids = self.retrieve_and_parse(&data, clients).await?;
                data.insert("top10_references".into(), json!(ids));
                data.insert("retrieved_examples".into(), json!([]));
            }
            _ => anyhow::bail!("Unknown retrieval_setting: {}", setting),
        }

        Ok(data)
    }

    fn load_manual_references(&self) -> Result<(Vec<String>, Value)> {
        let cfg = &self.task_config;
        let exp = &self.config.exp_config;

        if cfg.task_name == "diagram" {
            let path = exp
                .work_dir
                .join("data/PaperBananaBench/diagram/agent_selected_12.json");
            let raw = fs::read_to_string(&path)?;
            let all: Vec<Value> = serde_json::from_str(&raw)?;
            let examples: Vec<Value> = all.into_iter().take(10).collect();
            let ids: Vec<String> = examples
                .iter()
                .filter_map(|v| v.get("id").and_then(|id| id.as_str()).map(String::from))
                .collect();
            Ok((ids, json!(examples)))
        } else {
            Ok((vec![], json!([])))
        }
    }

    fn load_random_references(&self) -> Result<Vec<String>> {
        let cfg = &self.task_config;
        let exp = &self.config.exp_config;
        let path = exp
            .work_dir
            .join(format!("data/PaperBananaBench/{}/ref.json", cfg.task_name));
        let raw = fs::read_to_string(&path)?;
        let pool: Vec<Value> = serde_json::from_str(&raw)?;
        let mut ids: Vec<String> = pool
            .iter()
            .filter_map(|v| v.get("id").and_then(|id| id.as_str()).map(String::from))
            .collect();

        let mut rng = rand::rng();
        ids.shuffle(&mut rng);
        ids.truncate(10);
        Ok(ids)
    }

    async fn retrieve_and_parse(
        &self,
        data: &HashMap<String, Value>,
        clients: &ApiClients,
    ) -> Result<Vec<String>> {
        let cfg = &self.task_config;
        let exp = &self.config.exp_config;

        let content = data
            .get("content")
            .map(|v| v.to_string())
            .unwrap_or_default();
        let visual_intent = data
            .get("visual_intent")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let mut user_prompt = format!(
            "**Target Input**\n- {}: {}\n- {}: {}\n\n**Candidate Pool**\n",
            cfg.target_labels[0], visual_intent, cfg.target_labels[1], content
        );

        let path = exp
            .work_dir
            .join(format!("data/PaperBananaBench/{}/ref.json", cfg.task_name));
        let raw = fs::read_to_string(&path)?;
        let mut pool: Vec<Value> = serde_json::from_str(&raw)?;
        if let Some(limit) = cfg.ref_limit {
            pool.truncate(limit);
        }

        for (idx, item) in pool.iter().enumerate() {
            let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let vi = item
                .get("visual_intent")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let c = item.get("content").map(|v| v.to_string()).unwrap_or_default();
            user_prompt.push_str(&format!(
                "Candidate {} {}:\n- {}: {}\n- {}: {}\n- {}: {}\n\n",
                cfg.candidate_type,
                idx + 1,
                cfg.candidate_labels[0],
                id,
                cfg.candidate_labels[1],
                vi,
                cfg.candidate_labels[2],
                c,
            ));
        }

        user_prompt.push_str(&format!(
            "Now, based on the Target Input and the Candidate Pool, {}",
            cfg.instruction_suffix
        ));

        let contents = vec![ContentItem {
            content_type: "text".into(),
            text: Some(user_prompt),
            source: None,
            image_base64: None,
        }];

        let response_list = crate::generation_utils::call_gemini_with_retry_async(
            clients,
            &self.config.model_name,
            &contents,
            &self.config.system_prompt,
            exp.temperature,
            1,
            50000,
            5,
            30,
            None,
            "retriever_agent",
        )
        .await;

        if response_list.is_empty() {
            return Ok(vec![]);
        }

        let raw_response = response_list[0].trim();
        Ok(self.parse_retrieval_result(raw_response))
    }

    fn parse_retrieval_result(&self, raw_response: &str) -> Vec<String> {
        let cleaned = raw_response
            .replace("```json", "")
            .replace("```", "")
            .trim()
            .to_string();

        match serde_json::from_str::<Value>(&cleaned) {
            Ok(parsed) => {
                let key = if self.task_config.task_name == "plot" {
                    "top10_plots"
                } else {
                    "top10_diagrams"
                };
                parsed
                    .get(key)
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default()
            }
            Err(e) => {
                println!(
                    "Warning: Failed to parse retrieval result: {}. Raw: {}...",
                    e,
                    &raw_response[..raw_response.len().min(200)]
                );
                vec![]
            }
        }
    }
}

#[async_trait]
impl Agent for RetrieverAgent {
    async fn process(
        &self,
        data: HashMap<String, Value>,
        clients: &ApiClients,
    ) -> Result<HashMap<String, Value>> {
        let setting = self.config.exp_config.retrieval_setting.clone();
        self.process_with_setting(data, clients, &setting).await
    }
}

const DIAGRAM_RETRIEVER_AGENT_SYSTEM_PROMPT: &str = r#"
You are the Retrieval Agent for an academic diagram generation system. Your job is to select the Top 10 most relevant reference diagrams from a candidate pool that will serve as few-shot examples.

Selection criteria:
1. Match Research Topic: same domain (Agent & Reasoning, Vision & Perception, etc.)
2. Match Visual Intent: same diagram type (Framework, Pipeline, Module Detail, etc.)

Ranking: Same Topic AND Same Visual Intent > Same Visual Intent only > Avoid different visual intent.

Output strictly valid JSON: {"top10_diagrams": ["ref_1", "ref_25", ...]}
"#;

const PLOT_RETRIEVER_AGENT_SYSTEM_PROMPT: &str = r#"
You are the Retrieval Agent for an academic plot generation system. Your job is to select the Top 10 most relevant reference plots from a candidate pool that will serve as few-shot examples.

Selection criteria:
1. Match Data Characteristics: similar data types and dimensions
2. Match Visual Intent: same plot type (bar chart, line chart, scatter plot, etc.)

Ranking: Same Data Type AND Same Plot Type > Same Plot Type only > Avoid different plot type.

Output strictly valid JSON: {"top10_plots": ["ref_0", "ref_25", ...]}
"#;
