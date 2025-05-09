// Copyright (c) 2023 Microsoft Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

// Allow K8s YAML field names.
#![allow(non_snake_case)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// See ObjectMeta in the Kubernetes API reference.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ObjectMeta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    generateName: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    labels: Option<BTreeMap<String, String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<BTreeMap<String, String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub uid: Option<String>,
}

impl ObjectMeta {
    pub fn get_name(&self) -> String {
        if let Some(name) = &self.name {
            format!("^{}$", regex::escape(name))
        } else if let Some(generateName) = &self.generateName {
            // https://kubernetes.io/docs/concepts/overview/working-with-objects/names/#dns-subdomain-names
            format!("^{}[a-z0-9.-]*[a-z0-9]$", regex::escape(generateName))
        } else {
            String::new()
        }
    }

    pub fn get_namespace(&self) -> Option<String> {
        self.namespace.as_ref().cloned()
    }
}
