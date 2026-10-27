mod config;
mod console;
mod consolidation;
mod emotion;
mod llm;
mod memory;
mod mind;
mod persona;
mod profile;
mod prompt;
mod relationship;
mod session;
mod storage;
mod timefmt;
mod tracking;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rustyline::DefaultEditor;

use crate::{
    config::Config,
    llm::OpenAi,
    profile::{Paths, Profile},
    session::Session,
};

/// An AI companion with emotions and a memory that fades like a person's.
#[derive(Parser)]
#[command(version, about, after_help = format!("In-chat commands:\n{}", console::HELP))]
struct Cli {
    /// Where profiles and settings are stored [default: ~/.associative_memory_ai_companion]
    #[arg(long, env = "COMPANION_DATA_DIR", global = true)]
    data_dir: Option<PathBuf>,

    /// Which companion to talk to
    #[arg(short, long, default_value = "default", global = true)]
    profile: String,

    /// Use another model for this run only (switch for good with /model in the chat)
    #[arg(long, global = true)]
    model: Option<String>,

    /// Show inner thoughts and recalled memories
    #[arg(long, global = true)]
    debug: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Chat with a companion (default)
    Chat,
    /// List companions
    Profiles,
}

fn default_data_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("cannot determine the home directory")?;
    Ok(home.join(".associative_memory_ai_companion"))
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let data_dir = match cli.data_dir {
        Some(dir) => dir,
        None => default_data_dir()?,
    };
    let config = Config::load_or_create(&data_dir.join("config.json"))?;

    if let Some(Command::Profiles) = cli.command {
        return console::list_profiles(&data_dir, &config.embedding_model);
    }

    profile::validate_name(&cli.profile)?;
    let api_key =
        std::env::var("OPENAI_API_KEY").context("set the OPENAI_API_KEY environment variable")?;
    let llm = OpenAi::new(api_key, config.clone());

    let mut editor = DefaultEditor::new()?;
    let history = data_dir.join("history.txt");
    let _ = editor.load_history(&history);

    let paths = Paths::new(&data_dir, &cli.profile);
    let profile = if Profile::is_initialized(&paths) {
        Profile::load(&cli.profile, paths, &config.embedding_model)?
    } else {
        match console::setup(&llm, &config, &cli.profile, paths, &mut editor).await? {
            Some(profile) => profile,
            None => return Ok(()),
        }
    };

    let mut session = Session::new(llm, config, profile);
    if let Some(model) = cli.model {
        session.use_model_temporarily(model);
    }
    console::chat(session, editor, cli.debug, &history).await
}
