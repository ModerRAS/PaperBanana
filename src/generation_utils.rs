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

//! Utility functions for interacting with Gemini, Claude, OpenAI, and Doubao APIs.

use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Duration;

use crate::config::{self, ModelConfigFile};

/// Generic content item used across all providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentItem {
    #[serde(rename = "type")]
    pub content_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<ImageSource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_base64: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
}

/// Configuration for text model calls
#[derive(Debug, Clone)]
pub struct TextModelConfig {
    pub system_prompt: String,
    pub temperature: f64,
    pub candidate_num: usize,
    pub max_output_tokens: usize,
}

/// Configuration for image generation calls
#[derive(Debug, Clone)]
pub struct ImageGenConfig {
    pub size: String,
    pub quality: String,
    pub background: String,
    pub output_format: String,
    pub response_format: String,
    pub guidance_scale: f64,
    pub watermark: bool,
}

impl Default for ImageGenConfig {
    fn default() -> Self {
        Self {
            size: "1536x1024".to_string(),
            quality: "high".to_string(),
            background: "opaque".to_string(),
            output_format: "png".to_string(),
            response_format: "b64_json".to_string(),
            guidance_scale: 2.5,
            watermark: false,
        }
    }
}

/// API client holder for all providers
pub struct ApiClients {
    pub http_client: Client,
    pub google_api_key: Option<String>,
    pub anthropic_api_key: Option<String>,
    pub openai_api_key: Option<String>,
    pub doubao_api_key: Option<String>,
    pub doubao_base_url: String,
}

impl ApiClients {
    /// Initialize API clients from model config and environment variables
    pub fn new(model_config: &ModelConfigFile) -> Self {
        let http_client = Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .expect("Failed to create HTTP client");

        let google_api_key = Self::get_key(model_config, "api_keys", "google_api_key", "GOOGLE_API_KEY");
        let anthropic_api_key = Self::get_key(model_config, "api_keys", "anthropic_api_key", "ANTHROPIC_API_KEY");
        let openai_api_key = Self::get_key(model_config, "api_keys", "openai_api_key", "OPENAI_API_KEY");
        let doubao_api_key = Self::get_key(model_config, "api_keys", "doubao_api_key", "DOUBAO_API_KEY");
        let doubao_base_url = config::get_config_val(
            model_config,
            "doubao",
            "base_url",
            "DOUBAO_BASE_URL",
            "https://ark.cn-beijing.volces.com/api/v3",
        );

        if google_api_key.is_some() {
            println!("Initialized Gemini Client with API Key");
        } else {
            println!("Warning: Could not initialize Gemini Client. Missing credentials.");
        }
        if anthropic_api_key.is_some() {
            println!("Initialized Anthropic Client with API Key");
        } else {
            println!("Warning: Could not initialize Anthropic Client. Missing credentials.");
        }
        if openai_api_key.is_some() {
            println!("Initialized OpenAI Client with API Key");
        } else {
            println!("Warning: Could not initialize OpenAI Client. Missing credentials.");
        }
        if doubao_api_key.is_some() {
            println!("Initialized Doubao Client with API Key (Volcengine Ark SDK)");
        } else {
            println!("Warning: Could not initialize Doubao Client. Missing credentials.");
        }

        ApiClients {
            http_client,
            google_api_key,
            anthropic_api_key,
            openai_api_key,
            doubao_api_key,
            doubao_base_url,
        }
    }

    fn get_key(model_config: &ModelConfigFile, section: &str, key: &str, env_var: &str) -> Option<String> {
        let val = config::get_config_val(model_config, section, key, env_var, "");
        if val.is_empty() {
            None
        } else {
            Some(val)
        }
    }
}

// ======================== Format Converters ========================

/// Convert generic content list to Gemini API format
pub fn convert_to_gemini_parts(contents: &[ContentItem]) -> Vec<Value> {
    let mut parts = Vec::new();
    for item in contents {
        match item.content_type.as_str() {
            "text" => {
                if let Some(text) = &item.text {
                    parts.push(json!({"text": text}));
                }
            }
            "image" => {
                if let Some(source) = &item.source {
                    if source.source_type == "base64" {
                        if let (Some(data), Some(mime)) = (&source.data, &source.media_type) {
                            parts.push(json!({
                                "inline_data": {
                                    "mime_type": mime,
                                    "data": data
                                }
                            }));
                        }
                    }
                }
            }
            _ => {} // Unknown types are silently skipped
        }
    }
    parts
}

/// Convert generic content list to Claude API format (pass-through)
pub fn convert_to_claude_format(contents: &[ContentItem]) -> Vec<Value> {
    contents
        .iter()
        .map(|item| serde_json::to_value(item).unwrap_or(json!(null)))
        .collect()
}

/// Convert generic content list to OpenAI API format
pub fn convert_to_openai_format(contents: &[ContentItem]) -> Vec<Value> {
    let mut result = Vec::new();
    for item in contents {
        match item.content_type.as_str() {
            "text" => {
                if let Some(text) = &item.text {
                    result.push(json!({"type": "text", "text": text}));
                }
            }
            "image" => {
                if let Some(source) = &item.source {
                    if source.source_type == "base64" {
                        let media_type = source
                            .media_type
                            .as_deref()
                            .unwrap_or("image/jpeg");
                        let data = source.data.as_deref().unwrap_or("");
                        let data_url = format!("data:{};base64,{}", media_type, data);
                        result.push(json!({
                            "type": "image_url",
                            "image_url": {"url": data_url}
                        }));
                    }
                }
            }
            _ => {} // Unknown types are silently skipped
        }
    }
    result
}

// ======================== API Call Functions ========================

/// Call Gemini API with async retry logic
pub async fn call_gemini_with_retry_async(
    clients: &ApiClients,
    model_name: &str,
    contents: &[ContentItem],
    system_prompt: &str,
    temperature: f64,
    candidate_count: usize,
    max_output_tokens: usize,
    max_attempts: usize,
    retry_delay: u64,
    response_modalities: Option<Vec<String>>,
    error_context: &str,
) -> Vec<String> {
    let api_key = match &clients.google_api_key {
        Some(key) => key.clone(),
        None => {
            eprintln!("Gemini client was not initialized: missing Google API key.");
            return vec!["Error".to_string(); candidate_count];
        }
    };

    let mut result_list = Vec::new();
    let target_count = candidate_count;
    let effective_candidate_count = std::cmp::min(candidate_count, 8);

    for attempt in 0..max_attempts {
        let gemini_parts = convert_to_gemini_parts(contents);

        let mut gen_config = json!({
            "temperature": temperature,
            "candidateCount": effective_candidate_count,
            "maxOutputTokens": max_output_tokens,
        });

        if !system_prompt.is_empty() {
            // System instruction is set at the top level, not in gen config
        }

        if let Some(modalities) = &response_modalities {
            gen_config["responseModalities"] = json!(modalities);
        }

        let mut body = json!({
            "contents": [{"parts": gemini_parts}],
            "generationConfig": gen_config,
        });

        if !system_prompt.is_empty() {
            body["systemInstruction"] = json!({
                "parts": [{"text": system_prompt}]
            });
        }

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            model_name, api_key
        );

        match clients.http_client.post(&url).json(&body).send().await {
            Ok(resp) => {
                if let Ok(resp_json) = resp.json::<Value>().await {
                    if let Some(candidates) = resp_json["candidates"].as_array() {
                        for candidate in candidates {
                            if let Some(parts) = candidate["content"]["parts"].as_array() {
                                for part in parts {
                                    if let Some(text) = part["text"].as_str() {
                                        if !text.trim().is_empty() {
                                            result_list.push(text.to_string());
                                        }
                                    }
                                    if let Some(inline_data) = part.get("inlineData") {
                                        if let Some(data) = inline_data["data"].as_str() {
                                            result_list.push(data.to_string());
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if result_list.len() >= target_count {
                        result_list.truncate(target_count);
                        break;
                    }
                } else {
                    let context_msg = if error_context.is_empty() {
                        String::new()
                    } else {
                        format!(" for {}", error_context)
                    };
                    let current_delay = std::cmp::min(retry_delay * 2u64.pow(attempt as u32), 30);
                    eprintln!(
                        "Attempt {} for model {} failed{}: failed to parse response. Retrying in {} seconds...",
                        attempt + 1, model_name, context_msg, current_delay
                    );
                    if attempt < max_attempts - 1 {
                        tokio::time::sleep(Duration::from_secs(current_delay)).await;
                    }
                }
            }
            Err(e) => {
                let context_msg = if error_context.is_empty() {
                    String::new()
                } else {
                    format!(" for {}", error_context)
                };
                let current_delay = std::cmp::min(retry_delay * 2u64.pow(attempt as u32), 30);
                eprintln!(
                    "Attempt {} for model {} failed{}: {}. Retrying in {} seconds...",
                    attempt + 1, model_name, context_msg, e, current_delay
                );
                if attempt < max_attempts - 1 {
                    tokio::time::sleep(Duration::from_secs(current_delay)).await;
                } else {
                    eprintln!("Error: All {} attempts failed{}", max_attempts, context_msg);
                    return vec!["Error".to_string(); target_count];
                }
            }
        }
    }

    // Pad with errors if insufficient results
    while result_list.len() < target_count {
        result_list.push("Error".to_string());
    }
    result_list
}

/// Call Claude API with async retry logic
pub async fn call_claude_with_retry_async(
    clients: &ApiClients,
    model_name: &str,
    contents: &[ContentItem],
    system_prompt: &str,
    temperature: f64,
    candidate_num: usize,
    max_output_tokens: usize,
    max_attempts: usize,
    retry_delay: u64,
    error_context: &str,
) -> Vec<String> {
    let api_key = match &clients.anthropic_api_key {
        Some(key) => key.clone(),
        None => {
            eprintln!("Anthropic client was not initialized: missing API key.");
            return vec!["Error".to_string(); candidate_num];
        }
    };

    let mut response_text_list = Vec::new();

    // Validation phase - try to get first successful response
    let claude_contents = convert_to_claude_format(contents);

    for attempt in 0..max_attempts {
        let body = json!({
            "model": model_name,
            "max_tokens": max_output_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": claude_contents}],
            "system": system_prompt,
        });

        match clients
            .http_client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
        {
            Ok(resp) => {
                if let Ok(resp_json) = resp.json::<Value>().await {
                    if let Some(content) = resp_json["content"][0]["text"].as_str() {
                        response_text_list.push(content.to_string());
                        break;
                    }
                }
                let context_msg = if error_context.is_empty() {
                    String::new()
                } else {
                    format!(" for {}", error_context)
                };
                eprintln!(
                    "Validation attempt {} failed{}: empty response. Retrying in {} seconds...",
                    attempt + 1, context_msg, retry_delay
                );
                if attempt < max_attempts - 1 {
                    tokio::time::sleep(Duration::from_secs(retry_delay)).await;
                }
            }
            Err(e) => {
                let context_msg = if error_context.is_empty() {
                    String::new()
                } else {
                    format!(" for {}", error_context)
                };
                eprintln!(
                    "Validation attempt {} failed{}: {}. Retrying in {} seconds...",
                    attempt + 1, context_msg, e, retry_delay
                );
                if attempt < max_attempts - 1 {
                    tokio::time::sleep(Duration::from_secs(retry_delay)).await;
                }
            }
        }
    }

    if response_text_list.is_empty() {
        return vec!["Error".to_string(); candidate_num];
    }

    // Generate remaining candidates
    if candidate_num > 1 {
        let remaining = candidate_num - 1;
        let mut tasks = Vec::new();
        for _ in 0..remaining {
            let client = clients.http_client.clone();
            let api_key = api_key.clone();
            let model = model_name.to_string();
            let contents_clone = claude_contents.clone();
            let sys_prompt = system_prompt.to_string();
            let temp = temperature;
            let max_tokens = max_output_tokens;

            tasks.push(tokio::spawn(async move {
                let body = json!({
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temp,
                    "messages": [{"role": "user", "content": contents_clone}],
                    "system": sys_prompt,
                });
                match client
                    .post("https://api.anthropic.com/v1/messages")
                    .header("x-api-key", &api_key)
                    .header("anthropic-version", "2023-06-01")
                    .header("content-type", "application/json")
                    .json(&body)
                    .send()
                    .await
                {
                    Ok(resp) => {
                        if let Ok(resp_json) = resp.json::<Value>().await {
                            if let Some(content) = resp_json["content"][0]["text"].as_str() {
                                return content.to_string();
                            }
                        }
                        "Error".to_string()
                    }
                    Err(e) => {
                        eprintln!("Error generating candidate: {}", e);
                        "Error".to_string()
                    }
                }
            }));
        }

        for task in tasks {
            match task.await {
                Ok(result) => response_text_list.push(result),
                Err(e) => {
                    eprintln!("Error: task join failed: {}", e);
                    response_text_list.push("Error".to_string());
                }
            }
        }
    }

    response_text_list
}

/// Call OpenAI API with async retry logic
pub async fn call_openai_with_retry_async(
    clients: &ApiClients,
    model_name: &str,
    contents: &[ContentItem],
    system_prompt: &str,
    temperature: f64,
    candidate_num: usize,
    max_completion_tokens: usize,
    max_attempts: usize,
    retry_delay: u64,
    error_context: &str,
) -> Vec<String> {
    let api_key = match &clients.openai_api_key {
        Some(key) => key.clone(),
        None => {
            eprintln!("OpenAI client was not initialized: missing API key.");
            return vec!["Error".to_string(); candidate_num];
        }
    };

    let mut response_text_list = Vec::new();
    let openai_contents = convert_to_openai_format(contents);

    for attempt in 0..max_attempts {
        let body = json!({
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": openai_contents}
            ],
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
        });

        match clients
            .http_client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
        {
            Ok(resp) => {
                if let Ok(resp_json) = resp.json::<Value>().await {
                    if let Some(content) =
                        resp_json["choices"][0]["message"]["content"].as_str()
                    {
                        response_text_list.push(content.to_string());
                        break;
                    }
                }
                let context_msg = if error_context.is_empty() {
                    String::new()
                } else {
                    format!(" for {}", error_context)
                };
                eprintln!(
                    "Validation attempt {} failed{}: Retrying in {} seconds...",
                    attempt + 1, context_msg, retry_delay
                );
                if attempt < max_attempts - 1 {
                    tokio::time::sleep(Duration::from_secs(retry_delay)).await;
                }
            }
            Err(e) => {
                let context_msg = if error_context.is_empty() {
                    String::new()
                } else {
                    format!(" for {}", error_context)
                };
                eprintln!(
                    "Validation attempt {} failed{}: {}. Retrying in {} seconds...",
                    attempt + 1, context_msg, e, retry_delay
                );
                if attempt < max_attempts - 1 {
                    tokio::time::sleep(Duration::from_secs(retry_delay)).await;
                }
            }
        }
    }

    if response_text_list.is_empty() {
        return vec!["Error".to_string(); candidate_num];
    }

    // Generate remaining candidates in parallel
    if candidate_num > 1 {
        let remaining = candidate_num - 1;
        let mut tasks = Vec::new();
        for _ in 0..remaining {
            let client = clients.http_client.clone();
            let api_key = api_key.clone();
            let model = model_name.to_string();
            let contents_clone = openai_contents.clone();
            let sys_prompt = system_prompt.to_string();
            let temp = temperature;
            let max_tokens = max_completion_tokens;

            tasks.push(tokio::spawn(async move {
                let body = json!({
                    "model": model,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": contents_clone}
                    ],
                    "temperature": temp,
                    "max_completion_tokens": max_tokens,
                });
                match client
                    .post("https://api.openai.com/v1/chat/completions")
                    .header("Authorization", format!("Bearer {}", api_key))
                    .header("content-type", "application/json")
                    .json(&body)
                    .send()
                    .await
                {
                    Ok(resp) => {
                        if let Ok(resp_json) = resp.json::<Value>().await {
                            if let Some(content) =
                                resp_json["choices"][0]["message"]["content"].as_str()
                            {
                                return content.to_string();
                            }
                        }
                        "Error".to_string()
                    }
                    Err(e) => {
                        eprintln!("Error generating candidate: {}", e);
                        "Error".to_string()
                    }
                }
            }));
        }
        for task in tasks {
            match task.await {
                Ok(result) => response_text_list.push(result),
                Err(e) => {
                    eprintln!("Error: task join failed: {}", e);
                    response_text_list.push("Error".to_string());
                }
            }
        }
    }

    response_text_list
}

/// Call Doubao (豆包) API with async retry logic
pub async fn call_doubao_with_retry_async(
    clients: &ApiClients,
    model_name: &str,
    contents: &[ContentItem],
    system_prompt: &str,
    temperature: f64,
    candidate_num: usize,
    max_output_tokens: usize,
    max_attempts: usize,
    retry_delay: u64,
    error_context: &str,
) -> Vec<String> {
    let api_key = match &clients.doubao_api_key {
        Some(key) => key.clone(),
        None => {
            return Err::<Vec<String>, _>(anyhow::anyhow!(
                "Doubao client was not initialized: missing Doubao API key. \
                 Please set DOUBAO_API_KEY in environment, or configure api_keys.doubao_api_key in configs/model_config.yaml."
            ))
            .unwrap_or_else(|e| {
                eprintln!("{}", e);
                vec!["Error".to_string(); candidate_num]
            });
        }
    };

    let mut response_text_list = Vec::new();
    let doubao_contents = convert_to_openai_format(contents);

    for attempt in 0..max_attempts {
        let body = json!({
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": doubao_contents}
            ],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        });

        let url = format!("{}/chat/completions", clients.doubao_base_url);
        match clients
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
        {
            Ok(resp) => {
                if let Ok(resp_json) = resp.json::<Value>().await {
                    if let Some(content) =
                        resp_json["choices"][0]["message"]["content"].as_str()
                    {
                        response_text_list.push(content.to_string());
                        break;
                    }
                }
                let context_msg = if error_context.is_empty() {
                    String::new()
                } else {
                    format!(" for {}", error_context)
                };
                let current_delay = std::cmp::min(retry_delay * 2u64.pow(attempt as u32), 30);
                eprintln!(
                    "Validation attempt {} failed{}: Retrying in {} seconds...",
                    attempt + 1, context_msg, current_delay
                );
                if attempt < max_attempts - 1 {
                    tokio::time::sleep(Duration::from_secs(current_delay)).await;
                }
            }
            Err(e) => {
                let context_msg = if error_context.is_empty() {
                    String::new()
                } else {
                    format!(" for {}", error_context)
                };
                let current_delay = std::cmp::min(retry_delay * 2u64.pow(attempt as u32), 30);
                eprintln!(
                    "Validation attempt {} failed{}: {}. Retrying in {} seconds...",
                    attempt + 1, context_msg, e, current_delay
                );
                if attempt < max_attempts - 1 {
                    tokio::time::sleep(Duration::from_secs(current_delay)).await;
                }
            }
        }
    }

    if response_text_list.is_empty() {
        return vec!["Error".to_string(); candidate_num];
    }

    // Generate remaining candidates
    if candidate_num > 1 {
        let remaining = candidate_num - 1;
        let mut tasks = Vec::new();
        for _ in 0..remaining {
            let client = clients.http_client.clone();
            let api_key = api_key.clone();
            let model = model_name.to_string();
            let contents_clone = doubao_contents.clone();
            let sys_prompt = system_prompt.to_string();
            let temp = temperature;
            let max_tokens = max_output_tokens;
            let base_url = clients.doubao_base_url.clone();

            tasks.push(tokio::spawn(async move {
                let body = json!({
                    "model": model,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": contents_clone}
                    ],
                    "temperature": temp,
                    "max_tokens": max_tokens,
                });
                let url = format!("{}/chat/completions", base_url);
                match client
                    .post(&url)
                    .header("Authorization", format!("Bearer {}", api_key))
                    .header("content-type", "application/json")
                    .json(&body)
                    .send()
                    .await
                {
                    Ok(resp) => {
                        if let Ok(resp_json) = resp.json::<Value>().await {
                            if let Some(content) =
                                resp_json["choices"][0]["message"]["content"].as_str()
                            {
                                return content.to_string();
                            }
                        }
                        "Error".to_string()
                    }
                    Err(e) => {
                        eprintln!("Error generating candidate: {}", e);
                        "Error".to_string()
                    }
                }
            }));
        }
        for task in tasks {
            match task.await {
                Ok(result) => response_text_list.push(result),
                Err(e) => {
                    eprintln!("Error: task join failed: {}", e);
                    response_text_list.push("Error".to_string());
                }
            }
        }
    }

    response_text_list
}

/// Call OpenAI Image Generation API with async retry logic
pub async fn call_openai_image_generation_with_retry_async(
    clients: &ApiClients,
    model_name: &str,
    prompt: &str,
    config: &ImageGenConfig,
    max_attempts: usize,
    retry_delay: u64,
    error_context: &str,
) -> Vec<String> {
    let api_key = match &clients.openai_api_key {
        Some(key) => key.clone(),
        None => {
            eprintln!("OpenAI client was not initialized: missing API key.");
            return vec!["Error".to_string()];
        }
    };

    for attempt in 0..max_attempts {
        let body = json!({
            "model": model_name,
            "prompt": prompt,
            "n": 1,
            "size": config.size,
            "quality": config.quality,
            "background": config.background,
            "output_format": config.output_format,
        });

        match clients
            .http_client
            .post("https://api.openai.com/v1/images/generations")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
        {
            Ok(resp) => {
                if let Ok(resp_json) = resp.json::<Value>().await {
                    if let Some(data) = resp_json["data"][0]["b64_json"].as_str() {
                        return vec![data.to_string()];
                    }
                    if let Some(url) = resp_json["data"][0]["url"].as_str() {
                        return vec![url.to_string()];
                    }
                }
                eprintln!("[Warning]: Failed to generate image via OpenAI, no data returned.");
                if attempt < max_attempts - 1 {
                    tokio::time::sleep(Duration::from_secs(retry_delay)).await;
                }
            }
            Err(e) => {
                let context_msg = if error_context.is_empty() {
                    String::new()
                } else {
                    format!(" for {}", error_context)
                };
                eprintln!(
                    "Attempt {} for OpenAI image generation model {} failed{}: {}. Retrying in {} seconds...",
                    attempt + 1, model_name, context_msg, e, retry_delay
                );
                if attempt < max_attempts - 1 {
                    tokio::time::sleep(Duration::from_secs(retry_delay)).await;
                } else {
                    eprintln!("Error: All {} attempts failed{}", max_attempts, context_msg);
                    return vec!["Error".to_string()];
                }
            }
        }
    }

    vec!["Error".to_string()]
}

/// Call Doubao Image Generation API with async retry logic
pub async fn call_doubao_image_generation_with_retry_async(
    clients: &ApiClients,
    model_name: &str,
    prompt: &str,
    config: &ImageGenConfig,
    max_attempts: usize,
    retry_delay: u64,
    error_context: &str,
) -> Vec<String> {
    let api_key = match &clients.doubao_api_key {
        Some(key) => key.clone(),
        None => {
            eprintln!("Doubao client was not initialized: missing Doubao API key.");
            return vec!["Error".to_string()];
        }
    };

    for attempt in 0..max_attempts {
        let body = json!({
            "model": model_name,
            "prompt": prompt,
            "size": config.size,
            "response_format": config.response_format,
            "guidance_scale": config.guidance_scale,
            "watermark": config.watermark,
        });

        let url = format!("{}/images/generations", clients.doubao_base_url);
        match clients
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
        {
            Ok(resp) => {
                if let Ok(resp_json) = resp.json::<Value>().await {
                    if let Some(data) = resp_json["data"][0]["b64_json"].as_str() {
                        return vec![data.to_string()];
                    }
                    if let Some(url) = resp_json["data"][0]["url"].as_str() {
                        return vec![url.to_string()];
                    }
                }
                eprintln!("[Warning]: Failed to generate image via Doubao, no data returned.");
                if attempt < max_attempts - 1 {
                    tokio::time::sleep(Duration::from_secs(retry_delay)).await;
                }
            }
            Err(e) => {
                let context_msg = if error_context.is_empty() {
                    String::new()
                } else {
                    format!(" for {}", error_context)
                };
                let current_delay = std::cmp::min(retry_delay * 2u64.pow(attempt as u32), 60);
                eprintln!(
                    "Attempt {} for Doubao image generation model {} failed{}: {}. Retrying in {} seconds...",
                    attempt + 1, model_name, context_msg, e, current_delay
                );
                if attempt < max_attempts - 1 {
                    tokio::time::sleep(Duration::from_secs(current_delay)).await;
                } else {
                    eprintln!("Error: All {} attempts failed{}", max_attempts, context_msg);
                    return vec!["Error".to_string()];
                }
            }
        }
    }

    vec!["Error".to_string()]
}

/// Unified text generation dispatcher that routes to the correct provider
pub async fn call_text_model_with_retry_async(
    clients: &ApiClients,
    model_name: &str,
    contents: &[ContentItem],
    system_prompt: &str,
    temperature: f64,
    candidate_num: usize,
    max_output_tokens: usize,
    max_attempts: usize,
    retry_delay: u64,
    error_context: &str,
) -> Result<Vec<String>> {
    if model_name.contains("gemini") {
        Ok(call_gemini_with_retry_async(
            clients,
            model_name,
            contents,
            system_prompt,
            temperature,
            candidate_num,
            max_output_tokens,
            max_attempts,
            retry_delay,
            None,
            error_context,
        )
        .await)
    } else if model_name.contains("doubao") {
        Ok(call_doubao_with_retry_async(
            clients,
            model_name,
            contents,
            system_prompt,
            temperature,
            candidate_num,
            max_output_tokens,
            max_attempts,
            retry_delay,
            error_context,
        )
        .await)
    } else if model_name.contains("claude") {
        Ok(call_claude_with_retry_async(
            clients,
            model_name,
            contents,
            system_prompt,
            temperature,
            candidate_num,
            max_output_tokens,
            max_attempts,
            retry_delay,
            error_context,
        )
        .await)
    } else if model_name.contains("gpt")
        || model_name.contains("o1")
        || model_name.contains("o3")
        || model_name.contains("o4")
    {
        Ok(call_openai_with_retry_async(
            clients,
            model_name,
            contents,
            system_prompt,
            temperature,
            candidate_num,
            max_output_tokens,
            max_attempts,
            retry_delay,
            error_context,
        )
        .await)
    } else {
        Err(anyhow::anyhow!(
            "Unsupported text model: {}. Model name must contain 'gemini', 'doubao', 'claude', 'gpt', 'o1', 'o3', or 'o4'.",
            model_name
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ======================== Format Converter Tests ========================

    #[test]
    fn test_openai_text_only() {
        let contents = vec![ContentItem {
            content_type: "text".to_string(),
            text: Some("Hello world".to_string()),
            source: None,
            image_base64: None,
        }];
        let result = convert_to_openai_format(&contents);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["type"], "text");
        assert_eq!(result[0]["text"], "Hello world");
    }

    #[test]
    fn test_openai_image_base64() {
        let contents = vec![ContentItem {
            content_type: "image".to_string(),
            text: None,
            source: Some(ImageSource {
                source_type: "base64".to_string(),
                media_type: Some("image/png".to_string()),
                data: Some("abc123".to_string()),
            }),
            image_base64: None,
        }];
        let result = convert_to_openai_format(&contents);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["type"], "image_url");
        assert_eq!(
            result[0]["image_url"]["url"],
            "data:image/png;base64,abc123"
        );
    }

    #[test]
    fn test_openai_image_default_media_type() {
        let contents = vec![ContentItem {
            content_type: "image".to_string(),
            text: None,
            source: Some(ImageSource {
                source_type: "base64".to_string(),
                media_type: None,
                data: Some("abc123".to_string()),
            }),
            image_base64: None,
        }];
        let result = convert_to_openai_format(&contents);
        assert!(result[0]["image_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/jpeg;base64,"));
    }

    #[test]
    fn test_openai_mixed_content() {
        let contents = vec![
            ContentItem {
                content_type: "text".to_string(),
                text: Some("Look at this image:".to_string()),
                source: None,
                image_base64: None,
            },
            ContentItem {
                content_type: "image".to_string(),
                text: None,
                source: Some(ImageSource {
                    source_type: "base64".to_string(),
                    media_type: Some("image/jpeg".to_string()),
                    data: Some("imgdata".to_string()),
                }),
                image_base64: None,
            },
            ContentItem {
                content_type: "text".to_string(),
                text: Some("What do you see?".to_string()),
                source: None,
                image_base64: None,
            },
        ];
        let result = convert_to_openai_format(&contents);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0]["type"], "text");
        assert_eq!(result[1]["type"], "image_url");
        assert_eq!(result[2]["type"], "text");
    }

    #[test]
    fn test_openai_empty_content() {
        let contents: Vec<ContentItem> = vec![];
        let result = convert_to_openai_format(&contents);
        assert!(result.is_empty());
    }

    #[test]
    fn test_openai_unknown_type_skipped() {
        let contents = vec![
            ContentItem {
                content_type: "video".to_string(),
                text: None,
                source: None,
                image_base64: None,
            },
            ContentItem {
                content_type: "text".to_string(),
                text: Some("hello".to_string()),
                source: None,
                image_base64: None,
            },
        ];
        let result = convert_to_openai_format(&contents);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["type"], "text");
    }

    #[test]
    fn test_openai_image_non_base64_source_skipped() {
        let contents = vec![ContentItem {
            content_type: "image".to_string(),
            text: None,
            source: Some(ImageSource {
                source_type: "url".to_string(),
                media_type: None,
                data: None,
            }),
            image_base64: None,
        }];
        let result = convert_to_openai_format(&contents);
        assert!(result.is_empty());
    }

    #[test]
    fn test_gemini_text_part() {
        let contents = vec![ContentItem {
            content_type: "text".to_string(),
            text: Some("Hello Gemini".to_string()),
            source: None,
            image_base64: None,
        }];
        let result = convert_to_gemini_parts(&contents);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["text"], "Hello Gemini");
    }

    #[test]
    fn test_gemini_image_part() {
        let img_data = BASE64.encode(b"\x89PNG\r\n\x1a\n");
        let contents = vec![ContentItem {
            content_type: "image".to_string(),
            text: None,
            source: Some(ImageSource {
                source_type: "base64".to_string(),
                media_type: Some("image/png".to_string()),
                data: Some(img_data),
            }),
            image_base64: None,
        }];
        let result = convert_to_gemini_parts(&contents);
        assert_eq!(result.len(), 1);
        assert!(result[0].get("inline_data").is_some());
    }

    #[test]
    fn test_gemini_empty_content() {
        let contents: Vec<ContentItem> = vec![];
        let result = convert_to_gemini_parts(&contents);
        assert!(result.is_empty());
    }

    #[test]
    fn test_gemini_unknown_type_skipped() {
        let contents = vec![
            ContentItem {
                content_type: "audio".to_string(),
                text: None,
                source: None,
                image_base64: None,
            },
            ContentItem {
                content_type: "text".to_string(),
                text: Some("hello".to_string()),
                source: None,
                image_base64: None,
            },
        ];
        let result = convert_to_gemini_parts(&contents);
        assert_eq!(result.len(), 1);
    }

    // ======================== Model Dispatcher Tests ========================

    #[tokio::test]
    async fn test_unsupported_model_raises() {
        let clients = ApiClients {
            http_client: Client::new(),
            google_api_key: None,
            anthropic_api_key: None,
            openai_api_key: None,
            doubao_api_key: None,
            doubao_base_url: String::new(),
        };
        let result = call_text_model_with_retry_async(
            &clients,
            "unknown-model-xyz",
            &[ContentItem {
                content_type: "text".to_string(),
                text: Some("hi".to_string()),
                source: None,
                image_base64: None,
            }],
            "sys",
            0.5,
            1,
            50000,
            5,
            5,
            "",
        )
        .await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unsupported text model"));
    }
}
