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

//! Configuration for experiments

use std::path::PathBuf;

use chrono::Local;
use serde::{Deserialize, Serialize};

/// Experiment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpConfig {
    pub dataset_name: String,
    pub task_name: String,
    pub split_name: String,
    pub temperature: f64,
    pub exp_mode: String,
    pub retrieval_setting: String,
    pub max_critic_rounds: usize,
    pub model_name: String,
    pub image_model_name: String,
    pub work_dir: PathBuf,
    pub timestamp: String,
    pub exp_name: String,
    pub result_dir: PathBuf,
}

/// YAML model config file structure
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ModelConfigFile {
    pub defaults: Option<DefaultsConfig>,
    pub api_keys: Option<ApiKeysConfig>,
    pub doubao: Option<DoubaoConfig>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct DefaultsConfig {
    pub model_name: Option<String>,
    pub image_model_name: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ApiKeysConfig {
    pub google_api_key: Option<String>,
    pub openai_api_key: Option<String>,
    pub anthropic_api_key: Option<String>,
    pub doubao_api_key: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct DoubaoConfig {
    pub base_url: Option<String>,
    pub image_model_name: Option<String>,
}

impl ExpConfig {
    /// Create a new ExpConfig with post-initialization logic matching the Python version
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dataset_name: String,
        task_name: String,
        split_name: String,
        exp_mode: String,
        retrieval_setting: String,
        max_critic_rounds: usize,
        model_name: String,
        work_dir: PathBuf,
    ) -> Self {
        let mut model_name = model_name;
        let mut image_model_name = String::new();

        // Fallback to yaml config if no model_name provided
        if model_name.is_empty() || image_model_name.is_empty() {
            let config_path = work_dir.join("configs").join("model_config.yaml");
            if config_path.exists() {
                if let Ok(contents) = std::fs::read_to_string(&config_path) {
                    if let Ok(config) = serde_yaml::from_str::<ModelConfigFile>(&contents) {
                        if let Some(defaults) = &config.defaults {
                            if model_name.is_empty() {
                                model_name = defaults.model_name.clone().unwrap_or_default();
                            }
                            if image_model_name.is_empty() {
                                image_model_name =
                                    defaults.image_model_name.clone().unwrap_or_default();
                            }
                        }
                    }
                }
            }
        }

        let timestamp = Local::now().format("%m%d_%H%M").to_string();
        let exp_name = format!(
            "{}_{}ret_{}_{}",
            timestamp, retrieval_setting, exp_mode, split_name
        );
        let result_dir = work_dir
            .join("results")
            .join(format!("{}_{}", dataset_name, task_name));

        // Create result_dir if it doesn't exist
        let _ = std::fs::create_dir_all(&result_dir);

        ExpConfig {
            dataset_name,
            task_name,
            split_name,
            temperature: 1.0,
            exp_mode,
            retrieval_setting,
            max_critic_rounds,
            model_name,
            image_model_name,
            work_dir,
            timestamp,
            exp_name,
            result_dir,
        }
    }
}

/// Get a configuration value from environment variable or YAML config file.
/// Priority: env var > yaml config > default
pub fn get_config_val(
    model_config: &ModelConfigFile,
    section: &str,
    key: &str,
    env_var: &str,
    default: &str,
) -> String {
    // Check environment variable first
    if let Ok(val) = std::env::var(env_var) {
        if !val.is_empty() {
            return val;
        }
    }

    // Check YAML config
    let yaml_val = match section {
        "api_keys" => model_config.api_keys.as_ref().and_then(|keys| match key {
            "google_api_key" => keys.google_api_key.clone(),
            "openai_api_key" => keys.openai_api_key.clone(),
            "anthropic_api_key" => keys.anthropic_api_key.clone(),
            "doubao_api_key" => keys.doubao_api_key.clone(),
            _ => None,
        }),
        "doubao" => model_config.doubao.as_ref().and_then(|d| match key {
            "base_url" => d.base_url.clone(),
            "image_model_name" => d.image_model_name.clone(),
            _ => None,
        }),
        "defaults" => model_config.defaults.as_ref().and_then(|d| match key {
            "model_name" => d.model_name.clone(),
            "image_model_name" => d.image_model_name.clone(),
            _ => None,
        }),
        _ => None,
    };

    if let Some(val) = yaml_val {
        if !val.is_empty() {
            return val;
        }
    }

    default.to_string()
}

/// Load model config from YAML file
pub fn load_model_config(work_dir: &std::path::Path) -> ModelConfigFile {
    let config_path = work_dir.join("configs").join("model_config.yaml");
    if config_path.exists() {
        if let Ok(contents) = std::fs::read_to_string(&config_path) {
            if let Ok(config) = serde_yaml::from_str::<ModelConfigFile>(&contents) {
                return config;
            }
        }
    }
    ModelConfigFile::default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_env_var_takes_precedence() {
        let config = ModelConfigFile::default();
        env::set_var("TEST_KEY_RUST", "from_env");
        let result = get_config_val(
            &config,
            "api_keys",
            "test_key",
            "TEST_KEY_RUST",
            "default_val",
        );
        assert_eq!(result, "from_env");
        env::remove_var("TEST_KEY_RUST");
    }

    #[test]
    fn test_default_when_no_env_or_config() {
        let config = ModelConfigFile::default();
        env::remove_var("NONEXISTENT_KEY_12345");
        let result = get_config_val(
            &config,
            "nonexistent_section",
            "nonexistent_key",
            "NONEXISTENT_KEY_12345",
            "my_default",
        );
        assert_eq!(result, "my_default");
    }

    #[test]
    fn test_empty_default() {
        let config = ModelConfigFile::default();
        env::remove_var("NONEXISTENT_KEY_67890");
        let result = get_config_val(&config, "no_section", "no_key", "NONEXISTENT_KEY_67890", "");
        assert_eq!(result, "");
    }

    #[test]
    fn test_yaml_config_lookup() {
        let config = ModelConfigFile {
            api_keys: Some(ApiKeysConfig {
                google_api_key: Some("yaml_google_key".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        env::remove_var("GOOGLE_API_KEY_TEST");
        let result = get_config_val(
            &config,
            "api_keys",
            "google_api_key",
            "GOOGLE_API_KEY_TEST",
            "",
        );
        assert_eq!(result, "yaml_google_key");
    }

    #[test]
    fn test_env_var_overrides_yaml() {
        let config = ModelConfigFile {
            api_keys: Some(ApiKeysConfig {
                google_api_key: Some("yaml_key".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        env::set_var("GOOGLE_API_KEY_TEST2", "env_key");
        let result = get_config_val(
            &config,
            "api_keys",
            "google_api_key",
            "GOOGLE_API_KEY_TEST2",
            "default",
        );
        assert_eq!(result, "env_key");
        env::remove_var("GOOGLE_API_KEY_TEST2");
    }
}
