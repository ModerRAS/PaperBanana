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

//! Main script to launch PaperVizAgent

use clap::Parser;
use serde_json::Value;
use std::collections::HashMap;

use paper_banana::config::{self, ExpConfig};
use paper_banana::generation_utils::ApiClients;
use paper_banana::processor::PaperVizProcessor;

/// PaperVizAgent processing script
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Name of the dataset to use
    #[arg(long, default_value = "PaperBananaBench")]
    dataset_name: String,

    /// Task type: diagram or plot
    #[arg(long, default_value = "diagram")]
    task_name: String,

    /// Split of the dataset to use
    #[arg(long, default_value = "test")]
    split_name: String,

    /// Name of the experiment mode
    #[arg(long, default_value = "dev")]
    exp_mode: String,

    /// Retrieval setting for planner agent
    #[arg(long, default_value = "auto")]
    retrieval_setting: String,

    /// Maximum number of critic rounds
    #[arg(long, default_value = "3")]
    max_critic_rounds: usize,

    /// Model name to use
    #[arg(long, default_value = "")]
    model_name: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let work_dir = std::env::current_dir()?;

    let exp_config = ExpConfig::new(
        args.dataset_name,
        args.task_name,
        args.split_name,
        args.exp_mode,
        args.retrieval_setting,
        args.max_critic_rounds,
        args.model_name,
        work_dir.clone(),
    );

    let base_path = work_dir.join("data").join(&exp_config.dataset_name);
    let input_filename = base_path
        .join(&exp_config.task_name)
        .join(format!("{}.json", exp_config.split_name));
    let output_filename = exp_config
        .result_dir
        .join(format!("{}.json", exp_config.exp_name));

    println!("Input file: {:?}", input_filename);
    println!("Output file: {:?}", output_filename);

    // Load input data
    let input_data: Vec<HashMap<String, Value>> = if input_filename.exists() {
        let content = std::fs::read_to_string(&input_filename)?;
        serde_json::from_str(&content)?
    } else {
        eprintln!(
            "Warning: Input file not found: {:?}. Running with empty data.",
            input_filename
        );
        Vec::new()
    };

    // Initialize API clients
    let model_config = config::load_model_config(&work_dir);
    let clients = ApiClients::new(&model_config);

    // Create processor
    let processor = PaperVizProcessor::new(exp_config);

    // Batch process documents
    let concurrent_num = 10;
    println!("Using max concurrency: {}", concurrent_num);

    let results = processor
        .process_queries_batch(input_data, &clients, concurrent_num, true)
        .await;

    // Save results
    println!(
        "Saving results (count: {}) to {:?}",
        results.len(),
        output_filename
    );
    let json_string = serde_json::to_string_pretty(&results)?;
    std::fs::write(&output_filename, json_string)?;

    println!("Processing completed.");
    Ok(())
}
