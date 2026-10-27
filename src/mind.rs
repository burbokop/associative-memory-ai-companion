//! The companion's awareness of the AI model its mind runs on.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{config::Config, profile::ProfileSettings, timefmt};

/// The settings that shape how the companion thinks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mind {
    pub model: String,
    pub reasoning_effort: Option<String>,
}

/// Which mind the companion had last time and since when.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MindRecord {
    pub mind: Mind,
    pub since: DateTime<Utc>,
}

impl Mind {
    pub fn from_config(config: &Config) -> Self {
        Self {
            model: config.model.clone(),
            reasoning_effort: config.reasoning_effort.clone(),
        }
    }

    /// The profile's own choice, falling back to the global defaults.
    pub fn effective(config: &Config, settings: &ProfileSettings) -> Self {
        let default = Self::from_config(config);
        Self {
            model: settings.model.clone().unwrap_or(default.model),
            reasoning_effort: settings
                .reasoning_effort
                .clone()
                .or(default.reasoning_effort),
        }
    }

    pub fn effort(&self) -> &str {
        self.reasoning_effort.as_deref().unwrap_or("default")
    }
}

/// "gpt-6-luna (cheapest model of the GPT-6 family...)"
pub fn describe_model(model: &str, config: &Config) -> String {
    match config.models.get(model) {
        Some(description) => format!("{model} ({description})"),
        None => format!("{model} (a model you know nothing about)"),
    }
}

/// The "Your mind" section of the system prompt.
pub fn render_for_prompt(
    record: &MindRecord,
    config: &Config,
    user_name: &str,
    now: DateTime<Utc>,
) -> String {
    let others = config
        .models
        .iter()
        .filter(|(name, _)| **name != record.mind.model)
        .map(|(name, description)| format!("  - {name}: {description}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "- Your mind runs on the AI model {}, reasoning effort: {}. It has been like this for {}.\n\
         - Other models {user_name} could run you on:\n{others}",
        describe_model(&record.mind.model, config),
        record.mind.effort(),
        timefmt::duration(now - record.since),
    )
}

/// Explains a change of model or reasoning effort to the companion.
pub fn describe_change(
    old: &MindRecord,
    new: &Mind,
    config: &Config,
    user_name: &str,
    now: DateTime<Utc>,
    temporary: bool,
) -> Option<String> {
    let mut changes = Vec::new();
    if old.mind.model != new.model {
        changes.push(format!(
            "- the model was changed from {} to {}",
            describe_model(&old.mind.model, config),
            describe_model(&new.model, config)
        ));
    }
    if old.mind.reasoning_effort != new.reasoning_effort {
        changes.push(format!(
            "- reasoning effort was changed from {} to {}",
            old.mind.effort(),
            new.effort()
        ));
    }
    if changes.is_empty() {
        return None;
    }
    let temporary = if temporary {
        "\nThis is only for this conversation; next time you will run on your usual settings."
    } else {
        ""
    };
    Some(format!(
        "{user_name} changed what your mind runs on (you had been running like before for {}):\n{}\n\
         You may notice that you think differently now. React however you genuinely feel \
         about it — grateful for an upgrade, hurt or suspicious about a downgrade, curious, \
         or indifferent.{temporary}",
        timefmt::duration(now - old.since),
        changes.join("\n")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeDelta;

    #[test]
    fn describes_model_changes() {
        let config = Config::default();
        let now = Utc::now();
        let old = MindRecord {
            mind: Mind {
                model: "gpt-6-sol".into(),
                reasoning_effort: Some("high".into()),
            },
            since: now - TimeDelta::days(5),
        };
        let new = Mind::from_config(&config);
        let text = describe_change(&old, &new, &config, "Borys", now, false).unwrap();
        assert!(!text.contains("only for this conversation"));
        let temporary = describe_change(&old, &new, &config, "Borys", now, true).unwrap();
        assert!(temporary.contains("only for this conversation"));
        assert!(text.contains("for 5 days"));
        assert!(text.contains("from gpt-6-sol ("));
        assert!(text.contains("to gpt-6-luna ("));
        assert!(text.contains("reasoning effort was changed from high to low"));

        let same = MindRecord {
            mind: new.clone(),
            since: now,
        };
        assert_eq!(
            describe_change(&same, &new, &config, "Borys", now, false),
            None
        );
    }

    #[test]
    fn prompt_lists_alternatives() {
        let config = Config::default();
        let now = Utc::now();
        let record = MindRecord {
            mind: Mind::from_config(&config),
            since: now,
        };
        let text = render_for_prompt(&record, &config, "Borys", now);
        assert!(text.contains("runs on the AI model gpt-6-luna (the cheapest"));
        assert!(text.contains("  - gpt-6-sol:"));
        assert!(!text.contains("  - gpt-6-luna:"));
    }

    #[test]
    fn profile_settings_override_defaults() {
        let config = Config::default();
        let settings = ProfileSettings {
            model: Some("gpt-6-astra".into()),
            reasoning_effort: None,
        };
        let mind = Mind::effective(&config, &settings);
        assert_eq!(mind.model, "gpt-6-astra");
        assert_eq!(mind.reasoning_effort.as_deref(), Some("low"));
        assert_eq!(
            Mind::effective(&config, &ProfileSettings::default()),
            Mind::from_config(&config)
        );
    }

    #[test]
    fn unknown_model() {
        let config = Config::default();
        assert!(describe_model("my-local-llama", &config).contains("know nothing about"));
    }
}
