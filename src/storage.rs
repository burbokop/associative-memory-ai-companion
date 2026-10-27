use std::{fs, path::Path};

use anyhow::{Context, Result};
use serde::{Serialize, de::DeserializeOwned};

/// Reads a JSON file. Returns `None` if the file does not exist.
pub fn load<T: DeserializeOwned>(path: &Path) -> Result<Option<T>> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    let value =
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    Ok(Some(value))
}

/// Writes a JSON file atomically: a crash never leaves a half-written file behind.
pub fn save<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let tmp = path.with_extension("json.tmp");
    let text = serde_json::to_string_pretty(value)?;
    fs::write(&tmp, text).with_context(|| format!("writing {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| format!("replacing {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_missing_file() {
        let dir = std::env::temp_dir().join(format!("amc-storage-{}", std::process::id()));
        let path = dir.join("a/b.json");
        assert_eq!(load::<Vec<u32>>(&path).unwrap(), None);
        save(&path, &vec![1u32, 2, 3]).unwrap();
        assert_eq!(load::<Vec<u32>>(&path).unwrap(), Some(vec![1, 2, 3]));
        fs::remove_dir_all(dir).unwrap();
    }
}
