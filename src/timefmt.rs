//! Human-readable time formatting for prompts and the console.

use chrono::{DateTime, Local, TimeDelta, Utc};

/// "5 minutes", "3 hours", "9 days", "2 months"...
pub fn duration(d: TimeDelta) -> String {
    let minutes = d.num_minutes().max(0);
    let (value, unit) = match minutes {
        0 => return "less than a minute".into(),
        m if m < 60 => (m, "minute"),
        m if m < 60 * 24 => (m / 60, "hour"),
        m if m < 60 * 24 * 14 => (m / (60 * 24), "day"),
        m if m < 60 * 24 * 60 => (m / (60 * 24 * 7), "week"),
        m if m < 60 * 24 * 365 => (m / (60 * 24 * 30), "month"),
        m => (m / (60 * 24 * 365), "year"),
    };
    format!("{value} {unit}{}", if value == 1 { "" } else { "s" })
}

/// "3 days ago"
pub fn ago(time: DateTime<Utc>, now: DateTime<Utc>) -> String {
    let d = now - time;
    if d.num_minutes() < 1 {
        "just now".into()
    } else {
        format!("{} ago", duration(d))
    }
}

/// Local wall-clock time with weekday, e.g. "Tue 27 Oct 2026, 14:03".
pub fn stamp(time: DateTime<Utc>) -> String {
    time.with_timezone(&Local)
        .format("%a %d %b %Y, %H:%M")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations() {
        assert_eq!(duration(TimeDelta::seconds(20)), "less than a minute");
        assert_eq!(duration(TimeDelta::minutes(1)), "1 minute");
        assert_eq!(duration(TimeDelta::minutes(150)), "2 hours");
        assert_eq!(duration(TimeDelta::days(9)), "9 days");
        assert_eq!(duration(TimeDelta::days(21)), "3 weeks");
        assert_eq!(duration(TimeDelta::days(100)), "3 months");
        assert_eq!(duration(TimeDelta::days(800)), "2 years");
    }
}
