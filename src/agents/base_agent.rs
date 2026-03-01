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

//! Base agent trait for all agents

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

use crate::config::ExpConfig;
use crate::generation_utils::ApiClients;

/// Base trait for all agents in the system
#[async_trait]
pub trait Agent: Send + Sync {
    /// Process the input data and return the result
    async fn process(
        &self,
        data: HashMap<String, Value>,
        clients: &ApiClients,
    ) -> Result<HashMap<String, Value>>;
}

/// Common agent configuration
#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub model_name: String,
    pub system_prompt: String,
    pub exp_config: ExpConfig,
}
