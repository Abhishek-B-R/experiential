//! Shared content capture, independent of destination and hosted tenancy policy.

pub(crate) mod budget;
pub(crate) mod collector;
pub(crate) mod delivery;
mod local;
mod local_payload;
mod local_store;
mod messages;
pub(crate) mod metrics;
mod projection;
pub(crate) mod python;
pub(crate) mod reasoning;
pub(crate) mod record;
pub(crate) mod response;
