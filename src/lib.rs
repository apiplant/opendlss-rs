//! Cross-platform, host-side pieces of the OpenDLSS-NR implementation.
//!
//! The GPU graph is intentionally capability-gated: its shader route depends
//! on NVIDIA Vulkan extensions which are unavailable on most Linux drivers.
//!
//! To run the network on an image, load a model directory and call [`render`]:
//!
//! ```no_run
//! use opendlss_nr::{load_image, render, save_image, Backend, Conditioning, Model};
//!
//! let model = Model::load(opendlss_nr::config::model_dir().expect("home directory"), false)?; // see `opendlss setup`
//! let image = load_image("in.png".as_ref())?;
//! let out = render(&model, &image, Conditioning::default(), Backend::Auto)?;
//! println!("{} via {} in {:.1} ms", out.device, out.backend, out.frame_ms);
//! save_image("out.png".as_ref(), &out.image)?;
//! # anyhow::Ok(())
//! ```

pub mod config;
pub mod cuda;
pub mod dll;
pub mod geometry;
pub mod graph;
pub mod model;
pub mod network;
pub mod portable;
pub mod render;
pub mod runtime;
pub mod selftest;
pub mod vulkan;
pub mod weights;

pub use model::Model;
pub use network::{Conditioning, Image, load_image, save_image};
pub use render::{Backend, Rendered, render};
