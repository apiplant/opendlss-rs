//! Where `opendlss setup` keeps the model: `$XDG_CONFIG_HOME/opendlss/models/nr`, falling back to
//! `~/.config/opendlss/models/nr` when `$XDG_CONFIG_HOME` is unset.

use std::path::PathBuf;

/// `$XDG_CONFIG_HOME/opendlss`, defaulting to `~/.config/opendlss`.
pub fn config_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join("opendlss"))
}

/// The portable model directory `opendlss setup` writes (`manifest.json` + `model/`).
pub fn model_dir() -> Option<PathBuf> {
    Some(config_dir()?.join("models").join("nr"))
}

/// Expands a leading `~/` the way a shell would; the setup prompt does not go through one.
pub fn expand_path(input: &str) -> PathBuf {
    match input.strip_prefix("~/") {
        Some(rest) => match std::env::var_os("HOME") {
            Some(home) => PathBuf::from(home).join(rest),
            None => PathBuf::from(input),
        },
        None => PathBuf::from(input),
    }
}
