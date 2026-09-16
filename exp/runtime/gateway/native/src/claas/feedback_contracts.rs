//! Explicit feedback protocol: authenticated scope is never accepted from request data.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct FeedbackRequest {
    pub application_id: String,
    pub feedback_id: String,
    pub response_id: Option<String>,
    pub episode_id: Option<String>,
    pub text: Option<String>,
    pub reward: Option<f64>,
    pub success: Option<bool>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct FinalizeEpisodeRequest {
    pub application_id: String,
    pub episode_id: String,
    pub response_ids: Vec<String>,
    pub status: EpisodeStatus,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum EpisodeStatus {
    Completed,
    Failed,
    Abandoned,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum FeedbackError {
    Invalid(&'static str),
    MissingEvidence,
    Conflict,
    Capacity,
    Storage,
}

impl FeedbackRequest {
    pub(crate) fn validate(&self) -> Result<(), FeedbackError> {
        identifier(&self.application_id)?;
        identifier(&self.feedback_id)?;
        if self.response_id.is_some() == self.episode_id.is_some() {
            return Err(FeedbackError::Invalid(
                "Feedback must target exactly one response_id or episode_id.",
            ));
        }
        if let Some(value) = &self.response_id {
            identifier(value)?;
        }
        if let Some(value) = &self.episode_id {
            identifier(value)?;
        }
        if self.text.is_none() && self.reward.is_none() && self.success.is_none() {
            return Err(FeedbackError::Invalid(
                "Feedback needs text, reward, or success.",
            ));
        }
        if let Some(text) = &self.text {
            if text.trim().is_empty() || text.len() > 65_536 {
                return Err(FeedbackError::Invalid(
                    "Feedback text must contain 1 to 65536 bytes.",
                ));
            }
        }
        if self
            .reward
            .is_some_and(|reward| !reward.is_finite() || !(-1.0..=1.0).contains(&reward))
        {
            return Err(FeedbackError::Invalid(
                "Feedback reward must be finite and between -1 and 1.",
            ));
        }
        if let (Some(reward), Some(success)) = (self.reward, self.success) {
            if reward != if success { 1.0 } else { 0.0 } {
                return Err(FeedbackError::Invalid(
                    "Paired feedback requires reward=1 for success or reward=0 for failure.",
                ));
            }
        }
        Ok(())
    }
}

impl FinalizeEpisodeRequest {
    pub(crate) fn validate(&self) -> Result<(), FeedbackError> {
        identifier(&self.application_id)?;
        identifier(&self.episode_id)?;
        if self.response_ids.is_empty() || self.response_ids.len() > 1_024 {
            return Err(FeedbackError::Invalid(
                "Episodes require 1 to 1024 explicit response IDs.",
            ));
        }
        let mut seen = std::collections::HashSet::new();
        for response in &self.response_ids {
            identifier(response)?;
            if !seen.insert(response) {
                return Err(FeedbackError::Invalid(
                    "Episode response IDs must be unique.",
                ));
            }
        }
        Ok(())
    }
}

fn identifier(value: &str) -> Result<(), FeedbackError> {
    if value.trim().is_empty() || value.len() > 512 {
        return Err(FeedbackError::Invalid(
            "Identifiers must contain 1 to 512 nonblank bytes.",
        ));
    }
    Ok(())
}
