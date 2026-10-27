//! One companion = one profile directory with all of its files.

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{
    emotion::Emotions,
    memory::{LongTerm, Message},
    mind::MindRecord,
    persona::{Persona, PersonaMeta},
    storage,
};

pub struct Paths {
    pub dir: PathBuf,
}

impl Paths {
    pub fn new(data_dir: &Path, profile: &str) -> Self {
        Self {
            dir: data_dir.join("profiles").join(profile),
        }
    }

    /// The only file meant to be edited by hand.
    pub fn persona(&self) -> PathBuf {
        self.dir.join("persona.json")
    }
    pub fn persona_meta(&self) -> PathBuf {
        self.dir.join("persona_meta.json")
    }
    pub fn state(&self) -> PathBuf {
        self.dir.join("state.json")
    }
    pub fn short_term(&self) -> PathBuf {
        self.dir.join("short_term.json")
    }
    pub fn long_term(&self) -> PathBuf {
        self.dir.join("long_term.json")
    }
    pub fn embeddings(&self) -> PathBuf {
        self.dir.join("embeddings.json")
    }
}

/// Everything about the relationship that is not memory or persona.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    pub user_name: String,
    pub met_at: DateTime<Utc>,
    pub last_interaction: Option<DateTime<Utc>>,
    pub emotions: Emotions,
    pub emotions_at: DateTime<Utc>,
    /// The model the companion ran on last time, to notice changes.
    #[serde(default)]
    pub mind: Option<MindRecord>,
}

#[derive(Serialize, Deserialize)]
struct EmbeddingStore {
    model: String,
    vectors: HashMap<u64, Vec<f32>>,
}

pub struct Profile {
    pub name: String,
    pub paths: Paths,
    pub persona: Persona,
    pub persona_meta: PersonaMeta,
    pub state: State,
    pub short_term: Vec<Message>,
    pub long_term: LongTerm,
    embedding_model: String,
}

impl Profile {
    pub fn is_initialized(paths: &Paths) -> bool {
        paths.state().exists() && paths.persona_meta().exists()
    }

    pub fn create(
        name: &str,
        paths: Paths,
        persona: Persona,
        persona_meta: PersonaMeta,
        state: State,
        embedding_model: &str,
    ) -> Result<Self> {
        let profile = Self {
            name: name.to_string(),
            paths,
            persona,
            persona_meta,
            state,
            short_term: Vec::new(),
            long_term: LongTerm::default(),
            embedding_model: embedding_model.to_string(),
        };
        storage::save(&profile.paths.persona(), &profile.persona)?;
        profile.save_persona_meta()?;
        profile.save()?;
        Ok(profile)
    }

    pub fn load(name: &str, paths: Paths, embedding_model: &str) -> Result<Self> {
        let required = |path: PathBuf| -> Result<PathBuf> {
            if path.exists() {
                Ok(path)
            } else {
                bail!("{} is missing", path.display())
            }
        };
        let persona = storage::load(&required(paths.persona())?)?.unwrap_or_default();
        let persona_meta =
            storage::load(&required(paths.persona_meta())?)?.context("persona metadata missing")?;
        let state = storage::load(&required(paths.state())?)?.context("state missing")?;
        let short_term = storage::load(&paths.short_term())?.unwrap_or_default();
        let mut long_term: LongTerm = storage::load(&paths.long_term())?.unwrap_or_default();

        // Vectors from another embedding model are not comparable: drop them,
        // they will be recomputed.
        if let Some(store) = storage::load::<EmbeddingStore>(&paths.embeddings())?
            && store.model == embedding_model
        {
            long_term.embeddings = store.vectors;
        }

        Ok(Self {
            name: name.to_string(),
            paths,
            persona,
            persona_meta,
            state,
            short_term,
            long_term,
            embedding_model: embedding_model.to_string(),
        })
    }

    pub fn save_persona_meta(&self) -> Result<()> {
        storage::save(&self.paths.persona_meta(), &self.persona_meta)
    }

    /// Saves state and memories (not the persona, which belongs to the user).
    pub fn save(&self) -> Result<()> {
        storage::save(&self.paths.state(), &self.state)?;
        storage::save(&self.paths.short_term(), &self.short_term)?;
        storage::save(&self.paths.long_term(), &self.long_term)?;
        storage::save(
            &self.paths.embeddings(),
            &EmbeddingStore {
                model: self.embedding_model.clone(),
                vectors: self.long_term.embeddings.clone(),
            },
        )
    }
}

pub fn validate_name(name: &str) -> Result<()> {
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        bail!("profile name must be non-empty and contain only letters, digits, '-' and '_'");
    }
    Ok(())
}

/// Names of all profiles in the data dir.
pub fn list(data_dir: &Path) -> Result<Vec<String>> {
    let dir = data_dir.join("profiles");
    let mut names = Vec::new();
    if dir.exists() {
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                names.push(entry.file_name().to_string_lossy().into_owned());
            }
        }
    }
    names.sort();
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        memory::{Speaker, tests::memory},
        persona::Author,
    };
    use serde_json::json;

    #[test]
    fn create_save_load_roundtrip() {
        let data = std::env::temp_dir().join(format!("amc-profile-{}", std::process::id()));
        let now = Utc::now();
        let persona = json!({"name": "Mira"}).as_object().unwrap().clone();
        let meta = PersonaMeta::new(&persona, Author::Companion, now);
        let state = State {
            user_name: "Borys".into(),
            met_at: now,
            last_interaction: None,
            emotions: Emotions::default(),
            emotions_at: now,
            mind: None,
        };
        let mut p =
            Profile::create("p1", Paths::new(&data, "p1"), persona, meta, state, "m1").unwrap();
        p.short_term.push(Message::new(Speaker::User, "hi", now));
        p.long_term.add(memory("x", &[], 0.5, now), vec![1.0, 0.0]);
        p.save().unwrap();

        let loaded = Profile::load("p1", Paths::new(&data, "p1"), "m1").unwrap();
        assert_eq!(loaded.short_term.len(), 1);
        assert_eq!(loaded.long_term.embeddings.len(), 1);
        assert!(Profile::is_initialized(&loaded.paths));

        let other_model = Profile::load("p1", Paths::new(&data, "p1"), "m2").unwrap();
        assert!(other_model.long_term.embeddings.is_empty());
        assert_eq!(list(&data).unwrap(), vec!["p1"]);
        fs::remove_dir_all(data).unwrap();
    }

    #[test]
    fn profile_names() {
        assert!(validate_name("work-buddy_2").is_ok());
        assert!(validate_name("../etc").is_err());
        assert!(validate_name("").is_err());
    }
}
