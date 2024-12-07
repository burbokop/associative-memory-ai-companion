use chrono::Utc;
use clap::CommandFactory;
use clap::FromArgMatches;
use clap::Parser;
use client::get_answer;
use compress::compress_memory;
use openai_api_rust::Auth;
use openai_api_rust::OpenAI;
use std::borrow::Borrow;
use std::cell::RefCell;
use std::io::Write;
use std::ops::Deref;
use std::process::exit;
use std::sync::Arc;
use std::sync::Mutex;
use types::Emotion;
use types::LongTermMemory;
use types::MemoryQuant;
use types::UserMessage;

mod client;
mod compress;
mod types;

#[derive(Parser)]
#[command(version, about, long_about = None)]
enum Args {
    Chat,
}

mod world {
    use clap::{Parser, Subcommand};

    #[derive(Subcommand)]
    pub(crate) enum Mem {
        Clear,
        ClearAll,
        Dump,
        Log,
        Compress {
            #[clap(long, short)]
            retention_amount: f32,
        },
    }

    #[derive(Parser)]
    #[command(version, about, long_about = None)]
    pub(crate) enum Command {
        #[command(subcommand)]
        Mem(Mem),
        Sys {
            v: Vec<String>,
        },
        Respond,
        Exit,
    }
}

fn exec_chat() {
    let user_name = whoami::username();
    let asis_name = String::from_utf8(
        std::fs::read(
            homedir::my_home()
                .unwrap()
                .unwrap()
                .join("./associative-memory-ai-companion/name"),
        )
        .unwrap(),
    )
    .unwrap();
    let system_name = "system".to_string();
    let auth = Auth::from_env().unwrap();
    let openai = OpenAI::new(auth, "https://api.openai.com/v1/");

    let save_path = {
        let exe_path = std::env::current_exe().unwrap();
        let exe_dir = exe_path.parent().unwrap();
        exe_dir.join("save.json")
    };

    let initial_mem = String::from_utf8(
        std::fs::read(
            homedir::my_home()
                .unwrap()
                .unwrap()
                .join("./associative-memory-ai-companion/initial_mem"),
        )
        .unwrap(),
    )
    .unwrap();

    let default_messages = vec![MemoryQuant::LongTermMemory(LongTermMemory {
        summary: format!(
            "My name is {}. Your name is {}. {}",
            user_name, asis_name, initial_mem
        ),
        time: Utc::now(),
        emotion: Emotion {
            emotion: "Sense of existence".to_string(),
            level: 1.,
        },
    })];

    let default_messages_len = default_messages.len();

    let messages: Arc<Mutex<RefCell<Vec<MemoryQuant>>>> =
        Arc::new(Mutex::new(RefCell::new(if save_path.exists() {
            serde_json::from_str(&std::fs::read_to_string(&save_path).unwrap()).unwrap()
        } else {
            default_messages
        })));

    let initial_messages_len = messages.lock().unwrap().deref().borrow().len();

    {
        let messages = Arc::downgrade(&messages);
        ctrlc::set_handler(move || {
            println!("\nworld: Saving into: {:?}...", &save_path);
            let messages = messages.upgrade().unwrap();
            let messages = messages.lock().unwrap();
            std::fs::write(
                &save_path,
                serde_json::to_string_pretty(&messages.try_borrow().unwrap().deref()).unwrap(),
            )
            .unwrap();
            println!("world: Exiting.");
            exit(0);
        })
        .expect("Error setting Ctrl-C handler");
    }

    loop {
        print!("{}: ", user_name);
        std::io::stdout().flush().unwrap();

        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();

        if input.starts_with("#") {
            let command = &input[1..input.len() - 1];

            let mut sw = vec!["#".to_string()];
            sw.append(&mut shellwords::split(command).unwrap());

            match <world::Command as CommandFactory>::command().try_get_matches_from(sw) {
                Ok(mut matches) => {
                    match world::Command::from_arg_matches_mut(&mut matches).unwrap() {
                        world::Command::Mem(mem) => match mem {
                            world::Mem::Clear => {
                                let messages = messages.lock();
                                let drained_count = messages
                                    .unwrap()
                                    .borrow_mut()
                                    .drain(initial_messages_len..)
                                    .count();
                                println!("world: {} messages cleared", drained_count);
                            }
                            world::Mem::ClearAll => {
                                let messages = messages.lock();
                                let drained_count = messages
                                    .unwrap()
                                    .borrow_mut()
                                    .drain(default_messages_len..)
                                    .count();
                                println!("world: {} messages cleared", drained_count);
                            }
                            world::Mem::Dump => {
                                let messages = messages.lock();
                                println!("world: Messages: {:#?}", messages.unwrap().borrow());
                            }
                            world::Mem::Log => {
                                let messages = messages.lock();
                                for message in messages.unwrap().deref().borrow().iter() {
                                    match message {
                                        MemoryQuant::User(message) => {
                                            println!("{}: {}", user_name, message.content)
                                        }
                                        MemoryQuant::Assistant(message) => println!(
                                            "{}: {}, (e: {:?})",
                                            asis_name, message.content, message.emotion
                                        ),
                                        MemoryQuant::System(message) => {
                                            println!("{}: {}", system_name, message.content)
                                        }
                                        MemoryQuant::LongTermMemory(mem) => println!(
                                            "{}: {}, (e: {:?})",
                                            system_name, mem.summary, mem.emotion
                                        ),
                                    }
                                }
                            }
                            world::Mem::Compress { retention_amount } => {
                                let messages = messages.lock().unwrap();
                                let messages = messages.deref().borrow();
                                compress_memory(messages.clone(), retention_amount);
                                println!("world: Memory comressed");
                            }
                        },
                        world::Command::Exit => {
                            println!("world: Exiting.");
                            exit(0)
                        }
                        world::Command::Sys { v } => {
                            let messages = messages.lock().unwrap();
                            messages.borrow_mut().push(MemoryQuant::User(UserMessage {
                                content: v.join(" "),
                                time: Utc::now(),
                            }));
                            println!("world: system message sent")
                        }
                        world::Command::Respond => {
                            let messages = messages.lock().unwrap();
                            let (answer, tokens_usage) =
                                get_answer(&openai, messages.deref().borrow().clone(), 100);
                            println!(
                                "{}: {}, (t: {}, e: {:?})",
                                asis_name, answer.content, tokens_usage, answer.emotion
                            );
                            messages.borrow_mut().push(MemoryQuant::Assistant(answer));
                        }
                    }
                }
                Err(err) => println!("world: {}", err),
            }
        } else {
            let input = input
                .trim_matches(|c: char| c == '\n' || c.is_whitespace())
                .to_owned();
            let messages = messages.lock().unwrap();

            messages.borrow_mut().push(MemoryQuant::User(UserMessage {
                content: input,
                time: Utc::now(),
            }));

            let (answer, tokens_usage) =
                get_answer(&openai, messages.deref().borrow().clone(), 100);
            println!(
                "{}: {}, (t: {}, e: {:?})",
                asis_name, answer.content, tokens_usage, answer.emotion
            );
            messages.borrow_mut().push(MemoryQuant::Assistant(answer));
        }
    }
}

fn main() {
    match Args::parse() {
        Args::Chat => exec_chat(),
    }
}
