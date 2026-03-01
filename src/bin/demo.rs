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

//! CLI demo for PaperBanana
//!
//! Generates academic illustration candidates from method content and figure caption.
//! This is the command-line equivalent of the interactive Streamlit demo.

use clap::{Parser, Subcommand};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;

use paper_banana::config::{self, ExpConfig};
use paper_banana::generation_utils::ApiClients;
use paper_banana::processor::PaperVizProcessor;

/// PaperBanana interactive demo
#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "PaperBanana CLI demo for academic illustration generation"
)]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,

    /// Method section content (Markdown recommended)
    #[arg(long)]
    method: Option<String>,

    /// Figure caption
    #[arg(long)]
    caption: Option<String>,

    /// Task type: diagram or plot
    #[arg(long, default_value = "diagram")]
    task_name: String,

    /// Pipeline mode
    #[arg(long, default_value = "dev_full")]
    exp_mode: String,

    /// Retrieval setting: auto, manual, random, or none
    #[arg(long, default_value = "auto")]
    retrieval_setting: String,

    /// Number of candidates to generate
    #[arg(long, default_value = "1")]
    num_candidates: usize,

    /// Maximum number of critic rounds
    #[arg(long, default_value = "3")]
    max_critic_rounds: usize,

    /// Model name to use (defaults to model_config.yaml)
    #[arg(long, default_value = "")]
    model_name: String,

    /// Output directory for results
    #[arg(long, default_value = "results/demo")]
    output_dir: PathBuf,

    /// Aspect ratio hint (e.g., "16:9", "3:2")
    #[arg(long, default_value = "16:9")]
    aspect_ratio: String,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Generate illustration candidates from method content and caption
    Generate {
        /// Method section content (reads from stdin if not provided)
        #[arg(long)]
        method: Option<String>,
        /// Figure caption
        #[arg(long)]
        caption: Option<String>,
    },
    /// Refine an existing image with edit instructions
    Refine {
        /// Path to the image to refine
        #[arg(long)]
        image_path: PathBuf,
        /// Edit instructions
        #[arg(long)]
        instructions: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let work_dir = std::env::current_dir()?;

    // Load model config
    let model_config = config::load_model_config(&work_dir);
    let clients = ApiClients::new(&model_config);

    match &args.command {
        Some(Command::Refine {
            image_path,
            instructions,
        }) => {
            run_refine(image_path, instructions, &args, &clients, &work_dir).await?;
        }
        Some(Command::Generate { method, caption }) => {
            let method_content = method
                .as_deref()
                .or(args.method.as_deref())
                .map(str::to_string)
                .unwrap_or_else(read_from_stdin_if_tty);
            let caption_text = caption
                .as_deref()
                .or(args.caption.as_deref())
                .map(str::to_string)
                .unwrap_or_default();

            if method_content.is_empty() {
                eprintln!("Error: method content is required. Use --method or provide via stdin.");
                eprintln!(
                    "Usage: paper_banana_demo generate --method '<content>' --caption '<caption>'"
                );
                std::process::exit(1);
            }

            run_generate(&method_content, &caption_text, &args, &clients, &work_dir).await?;
        }
        None => {
            let method_content = args
                .method
                .as_deref()
                .map(str::to_string)
                .unwrap_or_else(read_from_stdin_if_tty);
            let caption_text = args.caption.clone().unwrap_or_default();

            if method_content.is_empty() {
                eprintln!("Error: method content is required. Use --method or provide via stdin.");
                eprintln!("Usage: paper_banana_demo --method '<content>' --caption '<caption>'");
                std::process::exit(1);
            }

            run_generate(&method_content, &caption_text, &args, &clients, &work_dir).await?;
        }
    }

    Ok(())
}

/// Run the generation pipeline
async fn run_generate(
    method_content: &str,
    caption: &str,
    args: &Args,
    clients: &ApiClients,
    work_dir: &std::path::Path,
) -> anyhow::Result<()> {
    println!("🍌 PaperBanana Demo - Generate Candidates");
    println!("  Task:        {}", args.task_name);
    println!("  Mode:        {}", args.exp_mode);
    println!("  Retrieval:   {}", args.retrieval_setting);
    println!("  Candidates:  {}", args.num_candidates);
    println!("  Critic rounds: {}", args.max_critic_rounds);
    println!();

    // Prepare output directory
    std::fs::create_dir_all(&args.output_dir)?;

    // Create experiment config
    let exp_config = ExpConfig::new(
        "Demo".to_string(),
        args.task_name.clone(),
        "demo".to_string(),
        args.exp_mode.clone(),
        args.retrieval_setting.clone(),
        args.max_critic_rounds,
        args.model_name.clone(),
        work_dir.to_path_buf(),
    );

    let processor = PaperVizProcessor::new(exp_config);

    // Build input data list (one entry per candidate)
    let data_list: Vec<HashMap<String, Value>> = (0..args.num_candidates)
        .map(|i| {
            let mut d: HashMap<String, Value> = HashMap::new();
            d.insert(
                "filename".to_string(),
                json!(format!("demo_candidate_{}", i)),
            );
            d.insert("caption".to_string(), json!(caption));
            d.insert("content".to_string(), json!(method_content));
            d.insert("visual_intent".to_string(), json!(caption));
            d.insert(
                "additional_info".to_string(),
                json!({"rounded_ratio": args.aspect_ratio}),
            );
            d.insert(
                "max_critic_rounds".to_string(),
                json!(args.max_critic_rounds),
            );
            d.insert("candidate_id".to_string(), json!(i));
            d
        })
        .collect();

    let total = data_list.len();
    println!("Generating {} candidate(s)...", total);

    let results = processor
        .process_queries_batch(data_list, clients, args.num_candidates.min(10), false)
        .await;

    println!("\n✅ Generated {}/{} candidates", results.len(), total);

    // Save results
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
    let output_path = args
        .output_dir
        .join(format!("candidates_{}.json", timestamp));

    let json_str = serde_json::to_string_pretty(&results)?;
    std::fs::write(&output_path, &json_str)?;
    println!("Results saved to: {:?}", output_path);

    // Save individual images if present
    let task = args.task_name.to_lowercase();
    let image_fields = [
        format!("target_{}_desc0_base64_jpg", task),
        format!("target_{}_stylist_desc0_base64_jpg", task),
        format!("vanilla_{}_base64_jpg", task),
        format!("polished_{}_base64_jpg", task),
    ];

    let mut saved_images = 0;
    for (idx, result) in results.iter().enumerate() {
        // Check eval_image_field first
        let field = result
            .get("eval_image_field")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let img_b64 = if !field.is_empty() {
            result.get(field).and_then(|v| v.as_str())
        } else {
            // Try known fields
            image_fields
                .iter()
                .find_map(|f| result.get(f).and_then(|v| v.as_str()))
        };

        if let Some(b64) = img_b64 {
            use base64::Engine;
            if let Ok(bytes) =
                base64::engine::general_purpose::STANDARD.decode(b64.trim_end_matches('\n'))
            {
                let img_path = args
                    .output_dir
                    .join(format!("candidate_{}_{}.jpg", idx, timestamp));
                std::fs::write(&img_path, &bytes)?;
                println!("  Image saved: {:?}", img_path);
                saved_images += 1;
            }
        }
    }

    if saved_images == 0 {
        println!("  Note: No image fields found in results (API keys may be missing).");
    }

    Ok(())
}

/// Run the image refinement pipeline
async fn run_refine(
    image_path: &PathBuf,
    instructions: &str,
    args: &Args,
    clients: &ApiClients,
    work_dir: &std::path::Path,
) -> anyhow::Result<()> {
    println!("🍌 PaperBanana Demo - Refine Image");
    println!("  Image:        {:?}", image_path);
    println!("  Instructions: {}", instructions);
    println!();

    // Load and encode the image
    let image_bytes = std::fs::read(image_path)?;
    use base64::Engine;
    let image_b64 = base64::engine::general_purpose::STANDARD.encode(&image_bytes);

    let ext = image_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("jpg")
        .to_lowercase();
    let media_type = if ext == "png" {
        "image/png"
    } else {
        "image/jpeg"
    };

    // Prepare output directory
    std::fs::create_dir_all(&args.output_dir)?;

    // Create experiment config for polish mode
    let exp_config = ExpConfig::new(
        "Demo".to_string(),
        args.task_name.clone(),
        "demo".to_string(),
        "dev_polish".to_string(),
        args.retrieval_setting.clone(),
        args.max_critic_rounds,
        args.model_name.clone(),
        work_dir.to_path_buf(),
    );

    let processor = PaperVizProcessor::new(exp_config);

    // Build input data
    let mut data: HashMap<String, Value> = HashMap::new();
    let task = args.task_name.to_lowercase();
    let img_field = format!("target_{}_base64_jpg", task);
    data.insert("filename".to_string(), json!("demo_refine"));
    data.insert(img_field, json!(image_b64));
    data.insert(
        "image_media_type".to_string(),
        json!(media_type.to_string()),
    );
    data.insert("edit_instructions".to_string(), json!(instructions));
    data.insert(
        "additional_info".to_string(),
        json!({"rounded_ratio": args.aspect_ratio}),
    );

    println!("Refining image...");
    let result = processor.process_single_query(data, clients, false).await?;

    let polished_field = format!("polished_{}_base64_jpg", task);
    if let Some(b64) = result.get(&polished_field).and_then(|v| v.as_str()) {
        if let Ok(bytes) =
            base64::engine::general_purpose::STANDARD.decode(b64.trim_end_matches('\n'))
        {
            let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
            let out_path = args.output_dir.join(format!("refined_{}.jpg", timestamp));
            std::fs::write(&out_path, &bytes)?;
            println!("✅ Refined image saved to: {:?}", out_path);
        }
    } else {
        println!("Note: No refined image in result (API keys may be missing).");
    }

    // Save full result JSON
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
    let json_path = args
        .output_dir
        .join(format!("refine_result_{}.json", timestamp));
    std::fs::write(&json_path, serde_json::to_string_pretty(&result)?)?;
    println!("Full result saved to: {:?}", json_path);

    Ok(())
}

/// Read from stdin when not in a TTY (pipe mode); otherwise return empty string
fn read_from_stdin_if_tty() -> String {
    use std::io::{self, Read};
    // Only read stdin when it's piped (not interactive)
    if atty::is_stdin() {
        String::new()
    } else {
        let mut buf = String::new();
        let _ = io::stdin().read_to_string(&mut buf);
        buf
    }
}

/// Simple TTY detection helper module
mod atty {
    pub fn is_stdin() -> bool {
        // Check if stdin is a terminal (not a pipe)
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let fd = std::io::stdin().as_raw_fd();
            unsafe { libc_isatty(fd) }
        }
        #[cfg(not(unix))]
        {
            false
        }
    }

    #[cfg(unix)]
    unsafe fn libc_isatty(fd: i32) -> bool {
        extern "C" {
            fn isatty(fd: i32) -> i32;
        }
        isatty(fd) != 0
    }
}
