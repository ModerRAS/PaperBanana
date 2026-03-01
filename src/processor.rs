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

//! Processing pipeline of PaperVizAgent

use anyhow::Result;
use serde_json::{json, Value};
use std::collections::HashMap;

use crate::agents::base_agent::Agent;
use crate::agents::critic_agent::CriticAgent;
use crate::agents::planner_agent::PlannerAgent;
use crate::agents::polish_agent::PolishAgent;
use crate::agents::retriever_agent::RetrieverAgent;
use crate::agents::stylist_agent::StylistAgent;
use crate::agents::vanilla_agent::VanillaAgent;
use crate::agents::visualizer_agent::VisualizerAgent;
use crate::config::ExpConfig;
use crate::eval_toolkits::get_score_for_image_referenced;
use crate::generation_utils::ApiClients;

/// Main class for multimodal document processor
pub struct PaperVizProcessor {
    pub exp_config: ExpConfig,
    pub vanilla_agent: VanillaAgent,
    pub planner_agent: PlannerAgent,
    pub visualizer_agent: VisualizerAgent,
    pub stylist_agent: StylistAgent,
    pub critic_agent: CriticAgent,
    pub retriever_agent: RetrieverAgent,
    pub polish_agent: PolishAgent,
}

impl PaperVizProcessor {
    pub fn new(exp_config: ExpConfig) -> Self {
        let vanilla_agent = VanillaAgent::new(exp_config.clone());
        let planner_agent = PlannerAgent::new(exp_config.clone());
        let visualizer_agent = VisualizerAgent::new(exp_config.clone());
        let stylist_agent = StylistAgent::new(exp_config.clone());
        let critic_agent = CriticAgent::new(exp_config.clone());
        let retriever_agent = RetrieverAgent::new(exp_config.clone());
        let polish_agent = PolishAgent::new(exp_config.clone());

        PaperVizProcessor {
            exp_config,
            vanilla_agent,
            planner_agent,
            visualizer_agent,
            stylist_agent,
            critic_agent,
            retriever_agent,
            polish_agent,
        }
    }

    /// Run multi-round critic iteration
    async fn run_critic_iterations(
        &self,
        mut data: HashMap<String, Value>,
        task_name: &str,
        max_rounds: usize,
        source: &str,
        clients: &ApiClients,
    ) -> Result<HashMap<String, Value>> {
        let current_best_image_key = if source == "planner" {
            format!("target_{}_desc0_base64_jpg", task_name)
        } else {
            format!("target_{}_stylist_desc0_base64_jpg", task_name)
        };
        let mut current_best = current_best_image_key;

        for round_idx in 0..max_rounds {
            data.insert("current_critic_round".to_string(), json!(round_idx));
            data = self
                .critic_agent
                .process_with_options(data, clients, source)
                .await?;

            let critic_key = format!("target_{}_critic_suggestions{}", task_name, round_idx);
            let suggestions = data
                .get(&critic_key)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            if suggestions.trim() == "No changes needed." {
                println!(
                    "[Critic Round {}] No changes needed. Stopping iteration.",
                    round_idx
                );
                break;
            }

            data = self.visualizer_agent.process(data, clients).await?;

            let new_image_key = format!("target_{}_critic_desc{}_base64_jpg", task_name, round_idx);
            if data.contains_key(&new_image_key) {
                current_best = new_image_key;
                println!(
                    "[Critic Round {}] Completed iteration. Visualization SUCCESS.",
                    round_idx
                );
            } else {
                println!(
                    "[Critic Round {}] Visualization FAILED. Rolling back to: {}",
                    round_idx, current_best
                );
                break;
            }
        }

        data.insert("eval_image_field".to_string(), json!(current_best));
        Ok(data)
    }

    /// Complete processing pipeline for a single query
    pub async fn process_single_query(
        &self,
        mut data: HashMap<String, Value>,
        clients: &ApiClients,
        do_eval: bool,
    ) -> Result<HashMap<String, Value>> {
        let exp_mode = &self.exp_config.exp_mode;
        let task_name = self.exp_config.task_name.to_lowercase();
        let retrieval_setting = &self.exp_config.retrieval_setting;

        match exp_mode.as_str() {
            "vanilla" => {
                data = self.vanilla_agent.process(data, clients).await?;
                data.insert(
                    "eval_image_field".to_string(),
                    json!(format!("vanilla_{}_base64_jpg", task_name)),
                );
            }

            "dev_planner" => {
                data = self
                    .retriever_agent
                    .process_with_setting(data, clients, retrieval_setting)
                    .await?;
                data = self.planner_agent.process(data, clients).await?;
                data = self.visualizer_agent.process(data, clients).await?;
                data.insert(
                    "eval_image_field".to_string(),
                    json!(format!("target_{}_desc0_base64_jpg", task_name)),
                );
            }

            "dev_planner_stylist" => {
                data = self
                    .retriever_agent
                    .process_with_setting(data, clients, retrieval_setting)
                    .await?;
                data = self.planner_agent.process(data, clients).await?;
                data = self.stylist_agent.process(data, clients).await?;
                data = self.visualizer_agent.process(data, clients).await?;
                data.insert(
                    "eval_image_field".to_string(),
                    json!(format!("target_{}_stylist_desc0_base64_jpg", task_name)),
                );
            }

            "dev_planner_critic" | "demo_planner_critic" => {
                data = self
                    .retriever_agent
                    .process_with_setting(data, clients, retrieval_setting)
                    .await?;
                data = self.planner_agent.process(data, clients).await?;
                data = self.visualizer_agent.process(data, clients).await?;
                let max_rounds = data
                    .get("max_critic_rounds")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(3) as usize;
                data = self
                    .run_critic_iterations(data, &task_name, max_rounds, "planner", clients)
                    .await?;
            }

            "dev_full" | "demo_full" => {
                data = self
                    .retriever_agent
                    .process_with_setting(data, clients, retrieval_setting)
                    .await?;
                data = self.planner_agent.process(data, clients).await?;
                data = self.stylist_agent.process(data, clients).await?;
                data = self.visualizer_agent.process(data, clients).await?;
                let max_rounds = data
                    .get("max_critic_rounds")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(self.exp_config.max_critic_rounds as u64)
                    as usize;
                data = self
                    .run_critic_iterations(data, &task_name, max_rounds, "stylist", clients)
                    .await?;
            }

            "dev_polish" => {
                data = self.polish_agent.process(data, clients).await?;
                data.insert(
                    "eval_image_field".to_string(),
                    json!(format!("polished_{}_base64_jpg", task_name)),
                );
            }

            "dev_retriever" => {
                data = self
                    .retriever_agent
                    .process_with_setting(data, clients, retrieval_setting)
                    .await?;
            }

            _ => {
                return Err(anyhow::anyhow!("Unknown experiment name: {}", exp_mode));
            }
        }

        let should_eval = do_eval && !exp_mode.contains("demo") && exp_mode != "dev_retriever";

        if should_eval {
            get_score_for_image_referenced(
                &mut data,
                &task_name,
                &self.exp_config.model_name,
                &self.exp_config.work_dir,
                clients,
            )
            .await?;
        }

        Ok(data)
    }

    /// Batch process queries with concurrency support
    pub async fn process_queries_batch(
        &self,
        data_list: Vec<HashMap<String, Value>>,
        clients: &ApiClients,
        max_concurrent: usize,
        do_eval: bool,
    ) -> Vec<HashMap<String, Value>> {
        let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(max_concurrent));
        let mut results = Vec::new();

        // Process sequentially for simplicity (concurrent processing would require Arc<Self>)
        let total = data_list.len();
        for (idx, data) in data_list.into_iter().enumerate() {
            let _permit = semaphore.acquire().await.unwrap();
            match self.process_single_query(data, clients, do_eval).await {
                Ok(result) => {
                    println!("Processing {}/{} completed", idx + 1, total);
                    results.push(result);
                }
                Err(e) => {
                    eprintln!("Error processing query {}: {}", idx + 1, e);
                }
            }
        }

        results
    }
}
