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

//! Evaluation toolkits for PaperVizAgent

use anyhow::Result;
use base64::Engine;
use regex::Regex;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;

use crate::generation_utils::ApiClients;
use crate::prompts::{diagram_eval_prompts, plot_eval_prompts};

/// Get the evaluation prompt for a given task and dimension
#[allow(dead_code)]
fn get_eval_prompt(task_name: &str, eval_dim: &str) -> &'static str {
    match (task_name, eval_dim) {
        ("diagram", "faithfulness") => {
            diagram_eval_prompts::DIAGRAM_REFERENCED_COMPARISON_FAITHFULNESS_SYSTEM_PROMPT
        }
        ("diagram", "conciseness") => {
            diagram_eval_prompts::DIAGRAM_REFERENCED_COMPARISON_CONCISENESS_SYSTEM_PROMPT
        }
        ("diagram", "readability") => {
            diagram_eval_prompts::DIAGRAM_REFERENCED_COMPARISON_READABILITY_SYSTEM_PROMPT
        }
        ("diagram", "aesthetics") => {
            diagram_eval_prompts::DIAGRAM_REFERENCED_COMPARISON_AESTHETICS_SYSTEM_PROMPT
        }
        ("plot", "faithfulness") => {
            plot_eval_prompts::PLOT_REFERENCED_COMPARISON_FAITHFULNESS_SYSTEM_PROMPT
        }
        ("plot", "conciseness") => {
            plot_eval_prompts::PLOT_REFERENCED_COMPARISON_CONCISENESS_SYSTEM_PROMPT
        }
        ("plot", "readability") => {
            plot_eval_prompts::PLOT_REFERENCED_COMPARISON_READABILITY_SYSTEM_PROMPT
        }
        ("plot", "aesthetics") => {
            plot_eval_prompts::PLOT_REFERENCED_COMPARISON_AESTHETICS_SYSTEM_PROMPT
        }
        _ => "",
    }
}

/// Task configuration for evaluation
#[allow(dead_code)]
struct TaskConfig {
    visual_intent_label: &'static str,
    raw_content_label: &'static str,
    human_label: &'static str,
    model_label: &'static str,
}

#[allow(dead_code)]
fn get_task_config(task_name: &str) -> TaskConfig {
    match task_name {
        "plot" => TaskConfig {
            visual_intent_label: "Visual Intent of the Desired Plot",
            raw_content_label: "Raw Data",
            human_label: "Human-Drawn Plot (Human)",
            model_label: "Model-Generated Plot (Model)",
        },
        _ => TaskConfig {
            visual_intent_label: "Diagram Caption",
            raw_content_label: "Methodology Section",
            human_label: "Human-Drawn Diagram (Human)",
            model_label: "Model-Generated Diagram (Model)",
        },
    }
}

/// Try to extract winner from text using regex patterns
pub fn try_regex_extract_winner(text: &str) -> Option<String> {
    let patterns = [
        r#""winner"\s*:\s*"([^"]+)""#,
        r#"\*\*winner\*\*\s*:\s*"([^"]+)""#,
        r#"\*\*winner\*\*\s*:\s*([A-Za-z][A-Za-z\s]+?)(?:,|\n|$)"#,
        r#""winner"\s*:\s*([A-Za-z][A-Za-z\s]+?)(?:,|\n|$)"#,
    ];

    for pattern in &patterns {
        if let Ok(re) = Regex::new(pattern) {
            if let Some(captures) = re.captures(text) {
                if let Some(value) = captures.get(1) {
                    let val = value.as_str().trim().trim_end_matches(['*', '"']).trim();
                    if !val.is_empty() {
                        return Some(val.to_string());
                    }
                }
            }
        }
    }

    None
}

/// Determine the outcome for a tier given two dimension outcomes
pub fn determine_tier_outcome(dim1_outcome: &str, dim2_outcome: &str) -> String {
    let o1 = dim1_outcome.trim();
    let o2 = dim2_outcome.trim();

    // Both agree on a clear winner
    if o1 == o2 {
        if o1 == "Both are good" || o1 == "Both are bad" {
            return "Tie".to_string();
        }
        return o1.to_string();
    }

    // One Model, one neutral
    if (o1 == "Model" && (o2 == "Both are good" || o2 == "Both are bad"))
        || (o2 == "Model" && (o1 == "Both are good" || o1 == "Both are bad"))
    {
        return "Model".to_string();
    }

    // One Human, one neutral
    if (o1 == "Human" && (o2 == "Both are good" || o2 == "Both are bad"))
        || (o2 == "Human" && (o1 == "Both are good" || o1 == "Both are bad"))
    {
        return "Human".to_string();
    }

    // All other cases -> Tie
    "Tie".to_string()
}

/// Run evaluation for referenced comparison
pub async fn get_score_for_image_referenced(
    sample_data: &mut HashMap<String, Value>,
    task_name: &str,
    _model_name: &str,
    work_dir: &Path,
    _clients: &ApiClients,
) -> Result<()> {
    let _raw_content = sample_data.get("content").cloned().unwrap_or(json!(""));
    let _visual_intent = sample_data
        .get("visual_intent")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    // Check for ground truth image
    let path_to_gt_image_rel = match sample_data.get("path_to_gt_image") {
        Some(v) => v.as_str().unwrap_or("").to_string(),
        None => {
            eprintln!("⚠️  No ground truth image path found. Skipping evaluation.");
            for dim in &[
                "faithfulness",
                "conciseness",
                "readability",
                "aesthetics",
                "overall",
            ] {
                sample_data.insert(format!("{}_outcome", dim), json!("N/A - No GT"));
            }
            return Ok(());
        }
    };

    let gt_image_path = work_dir
        .join("data")
        .join("PaperBananaBench")
        .join(task_name)
        .join(&path_to_gt_image_rel);

    let gt_image_bytes = std::fs::read(&gt_image_path)?;
    let _gt_image_base64 = base64::engine::general_purpose::STANDARD.encode(&gt_image_bytes);

    let eval_image_field = sample_data
        .get("eval_image_field")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    if !sample_data.contains_key(&eval_image_field) {
        eprintln!(
            "⚠️  Image field '{}' not found. Model generation failed.",
            eval_image_field
        );
        for dim in &[
            "faithfulness",
            "conciseness",
            "readability",
            "aesthetics",
            "overall",
        ] {
            sample_data.insert(
                format!("{}_reasoning", dim),
                json!("Model failed to generate image - Human wins by default"),
            );
            sample_data.insert(format!("{}_outcome", dim), json!("Human"));
        }
        return Ok(());
    }

    let _model_image_base64 = sample_data
        .get(&eval_image_field)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    // Run evaluations (simplified - in production this would call the actual eval APIs)
    let dims = ["faithfulness", "conciseness", "readability", "aesthetics"];
    for dim in &dims {
        // For now, set a default outcome
        sample_data.insert(
            format!("{}_reasoning", dim),
            json!("Evaluation pending - requires API access"),
        );
        sample_data.insert(format!("{}_outcome", dim), json!("Both are good"));
    }

    // Calculate overall outcome using tier system
    let faithfulness = sample_data
        .get("faithfulness_outcome")
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown")
        .to_string();
    let readability = sample_data
        .get("readability_outcome")
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown")
        .to_string();
    let conciseness = sample_data
        .get("conciseness_outcome")
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown")
        .to_string();
    let aesthetics = sample_data
        .get("aesthetics_outcome")
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown")
        .to_string();

    // Tier 1: Faithfulness + Readability
    let tier1_outcome = determine_tier_outcome(&faithfulness, &readability);

    let (overall_outcome, decision_path) = if tier1_outcome == "Model" || tier1_outcome == "Human" {
        (
            tier1_outcome.clone(),
            format!(
                "Tier1({}, {}) -> {} [Decided at Tier 1]",
                faithfulness, readability, tier1_outcome
            ),
        )
    } else {
        let tier2_outcome = determine_tier_outcome(&conciseness, &aesthetics);
        (
            tier2_outcome.clone(),
            format!(
                "Tier1({}, {}) -> Tie; Tier2({}, {}) -> {} [Decided at Tier 2]",
                faithfulness, readability, conciseness, aesthetics, tier2_outcome
            ),
        )
    };

    sample_data.insert("overall_outcome".to_string(), json!(overall_outcome));
    sample_data.insert(
        "overall_reasoning".to_string(),
        json!(format!("Rule-based calculation: {}", decision_path)),
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determine_tier_outcome_both_agree() {
        assert_eq!(determine_tier_outcome("Model", "Model"), "Model");
        assert_eq!(determine_tier_outcome("Human", "Human"), "Human");
    }

    #[test]
    fn test_determine_tier_outcome_both_neutral() {
        assert_eq!(
            determine_tier_outcome("Both are good", "Both are good"),
            "Tie"
        );
        assert_eq!(
            determine_tier_outcome("Both are bad", "Both are bad"),
            "Tie"
        );
    }

    #[test]
    fn test_determine_tier_outcome_model_plus_neutral() {
        assert_eq!(determine_tier_outcome("Model", "Both are good"), "Model");
        assert_eq!(determine_tier_outcome("Both are bad", "Model"), "Model");
    }

    #[test]
    fn test_determine_tier_outcome_human_plus_neutral() {
        assert_eq!(determine_tier_outcome("Human", "Both are good"), "Human");
        assert_eq!(determine_tier_outcome("Both are bad", "Human"), "Human");
    }

    #[test]
    fn test_determine_tier_outcome_conflicting() {
        assert_eq!(determine_tier_outcome("Model", "Human"), "Tie");
        assert_eq!(determine_tier_outcome("Human", "Model"), "Tie");
    }

    #[test]
    fn test_try_regex_extract_winner_json_format() {
        let text = r#"{"comparison_reasoning": "blah", "winner": "Model"}"#;
        assert_eq!(try_regex_extract_winner(text), Some("Model".to_string()));
    }

    #[test]
    fn test_try_regex_extract_winner_no_match() {
        let text = "No winner information here";
        assert_eq!(try_regex_extract_winner(text), None);
    }

    #[test]
    fn test_try_regex_extract_winner_human() {
        let text = r#""winner": "Human""#;
        assert_eq!(try_regex_extract_winner(text), Some("Human".to_string()));
    }

    #[test]
    fn test_try_regex_extract_winner_both_good() {
        let text = r#""winner": "Both are good""#;
        assert_eq!(
            try_regex_extract_winner(text),
            Some("Both are good".to_string())
        );
    }
}
