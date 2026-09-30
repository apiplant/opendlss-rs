//! Cross-platform, host-side pieces of the OpenDLSS-NR implementation.
//!
//! The GPU graph is intentionally capability-gated: its shader route depends
//! on NVIDIA Vulkan extensions which are unavailable on most Linux drivers.

pub mod cuda;
pub mod dll;
pub mod geometry;
pub mod graph;
pub mod model;
pub mod network;
pub mod portable;
pub mod runtime;
pub mod selftest;
pub mod vulkan;
pub mod weights;
