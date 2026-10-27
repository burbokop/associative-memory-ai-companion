//! Interactive console: first-run setup, chat loop and slash commands.

use std::{io::Write as _, path::Path};

use anyhow::{Result, bail};
use chrono::Utc;
use colored::Colorize;
use rustyline::{DefaultEditor, error::ReadlineError};

use crate::{
    config::Config,
    emotion::Emotions,
    llm::Llm,
    memory::{Message, Speaker},
    persona::{self, Author, PersonaMeta},
    profile::{Paths, Profile, State},
    relationship,
    session::{Reply, Session},
    storage, timefmt,
};

pub const HELP: &str = "\
/emotions          how your companion feels right now
/persona           show the persona and who wrote each field
/relationship      how your companion sees your relationship (they will know you looked;
                   with /debug on you also see their private opinion and wishes)
/memories [query]  list long-term memories (strongest first), or search them
/facts             core facts that are never forgotten
/compact           consolidate the conversation into long-term memory now
/model             show the model your companion runs on and the models it knows
/model <name>      switch model; saved for this companion (add --temp for this run only)
/effort <level>    switch reasoning effort, e.g. low, medium, high (also takes --temp)
/debug             toggle showing inner thoughts, recalled memories, relationship changes
/help              this help
/exit              leave (also Ctrl-D); everything is saved after each message";

fn ask(editor: &mut DefaultEditor, prompt: &str) -> Result<String> {
    match editor.readline(prompt) {
        Ok(line) => Ok(line.trim().to_string()),
        Err(ReadlineError::Interrupted | ReadlineError::Eof) => bail!("setup cancelled"),
        Err(e) => Err(e.into()),
    }
}

fn default_user_name() -> String {
    whoami::realname()
        .ok()
        .filter(|n| !n.trim().is_empty())
        .or_else(|| whoami::username().ok())
        .unwrap_or_else(|| "friend".into())
}

/// Creates a profile on first start. Returns `None` if the user chose to write
/// the persona first and should come back afterwards.
pub async fn setup(
    llm: &impl Llm,
    config: &Config,
    name: &str,
    paths: Paths,
    editor: &mut DefaultEditor,
) -> Result<Option<Profile>> {
    println!(
        "{}",
        format!("Setting up a new companion (profile '{name}').").bold()
    );
    let default_name = default_user_name();
    let user_name = ask(
        editor,
        &format!("What should your companion call you? [{default_name}] "),
    )?;
    let user_name = if user_name.is_empty() {
        default_name
    } else {
        user_name
    };
    let now = Utc::now();

    let persona_path = paths.persona();
    let (persona, author) = match storage::load(&persona_path)? {
        Some(persona) => {
            if persona::is_blank(&persona) {
                bail!("{} is empty — fill it in first", persona_path.display());
            }
            println!("Using the persona you wrote in {}.", persona_path.display());
            (persona, Author::User)
        }
        None => {
            println!("Who should your companion be?");
            println!("  1) Let them choose who they are (you may give a hint)");
            println!("  2) I'll write the persona myself");
            if ask(editor, "> ")? == "2" {
                storage::save(&persona_path, &persona::template())?;
                println!(
                    "A template was written to {}.\nFill it in (you can add your own fields) \
                     and start again.",
                    persona_path.display()
                );
                return Ok(None);
            }
            let hint = ask(
                editor,
                "Any wishes about who they might be? (Enter to skip) ",
            )?;
            print!("{}", "They are deciding who they are…".dimmed());
            std::io::stdout().flush()?;
            let persona = persona::generate(llm, &user_name, Some(&hint)).await;
            clear_line();
            let persona = persona?;
            println!(
                "{} has chosen who they are. The persona is saved in {} — \
                 you may edit it, they will notice.",
                persona::name(&persona).bold(),
                persona_path.display()
            );
            (persona, Author::Companion)
        }
    };

    let meta = PersonaMeta::new(&persona, author, now);
    let state = State {
        user_name,
        met_at: now,
        last_interaction: None,
        emotions: Emotions::default(),
        emotions_at: now,
        mind: None,
    };
    Profile::create(name, paths, persona, meta, state, &config.embedding_model).map(Some)
}

fn clear_line() {
    print!("\r\x1b[2K");
    let _ = std::io::stdout().flush();
}

fn typing(name: &str) {
    print!("{}", format!("{name} is typing…").dimmed());
    let _ = std::io::stdout().flush();
}

fn print_thoughts(thoughts: &str) {
    println!("{}", format!("  💭 {thoughts}").dimmed().italic());
}

fn print_reply(name: &str, reply: &Reply, debug: bool) {
    if debug {
        for memory in &reply.recalled {
            println!("{}", format!("  🔎 recalled: {memory}").dimmed());
        }
        print_thoughts(&reply.thoughts);
    }
    println!("{}  {}", name.bold().magenta(), reply.emotions.render());
    println!("{}", reply.content);
    if debug && !reply.relationship_changes.is_empty() {
        println!(
            "{}",
            format!(
                "  💞 {}",
                relationship::render_changes(&reply.relationship_changes)
            )
            .dimmed()
        );
    }
    println!();
}

fn print_new_label(name: &str, label: &Option<String>) {
    if let Some(label) = label {
        println!(
            "{}",
            format!("💞 {name} now sees your relationship as: {label}").italic()
        );
    }
}

/// `/model` without arguments: current settings and the known models.
fn print_mind<L: Llm>(session: &Session<L>) {
    let source = if session.temporary_mind {
        "this run only"
    } else if session.profile.settings.model.is_some() {
        "chosen for this companion"
    } else {
        "default from config.json"
    };
    println!(
        "{} {} ({source}), reasoning effort {}",
        "Model:".bold(),
        session.mind.model,
        session.mind.effort()
    );
    println!("{}", "Known models (config.json → models):".bold());
    for (model, description) in &session.config.models {
        let marker = if *model == session.mind.model {
            "▶"
        } else {
            " "
        };
        println!("{marker} {model:<14} {}", description.dimmed());
    }
}

/// Shows the end of the previous conversation when resuming.
fn print_history(messages: &[&Message], name: &str, user: &str, debug: bool) {
    if messages.is_empty() {
        return;
    }
    println!("{}", "── earlier ──".dimmed());
    for m in messages {
        let stamp = format!("[{}]", timefmt::stamp(m.time)).dimmed();
        match m.speaker {
            Speaker::User => println!(
                "{stamp} {} {}\n",
                format!("{user}>").bold().cyan(),
                m.content
            ),
            Speaker::Companion => {
                if debug && let Some(thoughts) = &m.thoughts {
                    print_thoughts(thoughts);
                }
                let emotions = m
                    .emotions
                    .as_ref()
                    .map(Emotions::render)
                    .unwrap_or_default();
                println!("{stamp} {}  {emotions}", name.bold().magenta());
                println!("{}\n", m.content);
            }
            Speaker::Event => {}
        }
    }
    println!("{}", "── now ──".dimmed());
}

fn print_error(e: &anyhow::Error) {
    eprintln!("{} {e:#}", "error:".red().bold());
}

pub async fn chat<L: Llm>(
    mut session: Session<L>,
    mut editor: DefaultEditor,
    mut debug: bool,
    history: &Path,
) -> Result<()> {
    let name = session.companion_name();
    let user = session.user_name().to_string();
    println!(
        "{}",
        format!(
            "Chatting with {name} (profile '{}', model {}{}). Type /help for commands.",
            session.profile.name,
            session.mind.model,
            if session.temporary_mind {
                ", this run only"
            } else {
                ""
            }
        )
        .dimmed()
    );
    println!(
        "{}\n",
        relationship::render_summary(&session.profile.relationship)
    );

    print_history(
        &session.recent_messages(session.config.resume_history_messages),
        &name,
        &user,
        debug,
    );

    typing(&name);
    let opening = session.start().await;
    clear_line();
    match opening {
        Ok(Some(reply)) => print_reply(&name, &reply, debug),
        Ok(None) => {}
        Err(e) => print_error(&e),
    }

    let prompt = format!("{} ", format!("{user}>").bold().cyan());
    loop {
        let line = match editor.readline(&prompt) {
            Ok(line) => line,
            Err(ReadlineError::Interrupted | ReadlineError::Eof) => break,
            Err(e) => return Err(e.into()),
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let _ = editor.add_history_entry(line);

        if let Some(command) = line.strip_prefix('/') {
            let (command, argument) = command.split_once(' ').unwrap_or((command, ""));
            match command {
                "exit" | "quit" => break,
                "model" | "effort" => {
                    let mut words: Vec<&str> = argument.split_whitespace().collect();
                    let temporary = words.contains(&"--temp");
                    words.retain(|w| *w != "--temp");
                    let Some(value) = words.first().map(|w| w.to_string()) else {
                        print_mind(&session);
                        continue;
                    };
                    let mut new = session.mind.clone();
                    if command == "model" {
                        if !session.config.models.contains_key(&value) {
                            println!(
                                "{}",
                                format!(
                                    "note: {value} is not in the model catalog in config.json; \
                                     {name} will know nothing about it"
                                )
                                .yellow()
                            );
                        }
                        new.model = value;
                    } else {
                        new.reasoning_effort = Some(value);
                    }
                    typing(&name);
                    let result = session.switch_mind(new, !temporary).await;
                    clear_line();
                    match result {
                        Ok(reply) => {
                            println!(
                                "{}",
                                format!(
                                    "{name} now runs on {} (reasoning effort {}) — {}",
                                    session.mind.model,
                                    session.mind.effort(),
                                    if temporary {
                                        "for this run only".to_string()
                                    } else {
                                        format!("saved for {name}")
                                    }
                                )
                                .dimmed()
                            );
                            if let Some(reply) = reply {
                                print_reply(&name, &reply, debug);
                            }
                        }
                        Err(e) => print_error(&e),
                    }
                }
                "help" => println!("{HELP}"),
                "debug" => {
                    debug = !debug;
                    println!("debug {}", if debug { "on" } else { "off" });
                }
                "emotions" => println!("{}", session.profile.state.emotions.render()),
                "persona" => {
                    println!(
                        "{}",
                        session
                            .profile
                            .paths
                            .persona()
                            .display()
                            .to_string()
                            .dimmed()
                    );
                    println!(
                        "{}",
                        persona::render_for_prompt(
                            &session.profile.persona,
                            &session.profile.persona_meta
                        )
                    );
                }
                "facts" => {
                    for fact in &session.profile.long_term.core_facts {
                        println!("- {}", fact.fact);
                    }
                }
                "memories" => {
                    let list = if argument.trim().is_empty() {
                        Ok(session.memories_by_retention())
                    } else {
                        session.search_memories(argument.trim()).await
                    };
                    match list {
                        Ok(list) if list.is_empty() => println!("(no memories)"),
                        Ok(list) => {
                            let now = Utc::now();
                            for (score, memory) in list {
                                println!(
                                    "{} {}",
                                    format!("[{score:.2}]").dimmed(),
                                    memory.render(now)
                                );
                            }
                        }
                        Err(e) => print_error(&e),
                    }
                }
                "relationship" => {
                    println!(
                        "{}",
                        relationship::render_full(
                            &session.profile.relationship,
                            &session.profile.relationship_meta,
                            debug,
                        )
                    );
                    println!(
                        "{}",
                        session
                            .profile
                            .paths
                            .relationship()
                            .display()
                            .to_string()
                            .dimmed()
                    );
                    if let Err(e) = session.relationship_viewed(debug) {
                        print_error(&e);
                    }
                }
                "compact" => match session.maintain_memory(true).await {
                    Ok(r) => {
                        println!(
                            "{} messages → {} new memories; {} faded memories merged",
                            r.consolidated_messages, r.new_memories, r.faded_memories
                        );
                        print_new_label(&name, &r.new_relationship_label);
                    }
                    Err(e) => print_error(&e),
                },
                _ => println!("unknown command; try /help"),
            }
            continue;
        }

        typing(&name);
        let reply = session.user_says(line).await;
        clear_line();
        match reply {
            Ok(reply) => print_reply(&name, &reply, debug),
            Err(e) => {
                print_error(&e);
                continue;
            }
        }

        match session.maintain_memory(false).await {
            Ok(r) => {
                if debug && r.consolidated_messages > 0 {
                    println!(
                        "{}",
                        format!(
                            "  🧠 consolidated {} messages into {} memories, merged {} faded ones",
                            r.consolidated_messages, r.new_memories, r.faded_memories
                        )
                        .dimmed()
                    );
                }
                print_new_label(&name, &r.new_relationship_label);
            }
            // Short-term memory is kept, so this is retried after the next message.
            Err(e) => print_error(&e.context("memory consolidation failed")),
        }
    }

    let _ = editor.save_history(history);
    println!("{}", format!("{name} will remember this. Bye!").dimmed());
    Ok(())
}

/// `companion profiles`
pub fn list_profiles(data_dir: &Path, embedding_model: &str) -> Result<()> {
    let names = crate::profile::list(data_dir)?;
    if names.is_empty() {
        println!("No profiles yet in {}.", data_dir.display());
    }
    let now = Utc::now();
    for name in names {
        let paths = Paths::new(data_dir, &name);
        if !Profile::is_initialized(&paths) {
            println!("{name}: (not set up yet)");
            continue;
        }
        match Profile::load(&name, paths, embedding_model) {
            Ok(p) => println!(
                "{name}: {} — {} memories, last talked {}",
                persona::name(&p.persona).bold(),
                p.long_term.memories.len(),
                p.state
                    .last_interaction
                    .map_or("never".into(), |t| timefmt::ago(t, now))
            ),
            Err(e) => println!("{name}: {}", format!("{e:#}").red()),
        }
    }
    Ok(())
}
