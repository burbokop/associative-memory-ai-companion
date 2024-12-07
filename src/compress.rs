use crate::types::{AssistantMessage, LongTermMemory, MemoryQuant, SystemMessage, UserMessage};

/// retention_amount: value from 0..1. if value is 0 - none of memory is saved, if 1 - all memory is saved
pub(crate) fn compress_memory(
    messages: Vec<MemoryQuant>,
    retention_amount: f32,
) -> Vec<MemoryQuant> {
    enum Condition {
        User(UserMessage),
        System(SystemMessage),
    }

    struct Chunk {
        pub conditions: Vec<Condition>,
        pub responce: AssistantMessage,
    }

    fn split_into_chunks(messages: Vec<MemoryQuant>) -> (Vec<Chunk>, Vec<LongTermMemory>) {
        let mut chunks: Vec<Chunk> = Default::default();
        let mut long_term_memory: Vec<LongTermMemory> = Default::default();

        let mut current_conditions: Vec<Condition> = Default::default();

        for message in messages {
            match message {
                MemoryQuant::User(message) => current_conditions.push(Condition::User(message)),
                MemoryQuant::Assistant(message) => chunks.push(Chunk {
                    conditions: std::mem::take(&mut current_conditions),
                    responce: message,
                }),
                MemoryQuant::System(message) => current_conditions.push(Condition::System(message)),
                MemoryQuant::LongTermMemory(mem) => long_term_memory.push(mem),
            }
        }

        (chunks, long_term_memory)
    }

    let (mut chunks, long_term_memory) = split_into_chunks(messages);

    chunks.sort_by(|a, b| {
        a.responce
            .emotion
            .level
            .partial_cmp(&b.responce.emotion.level)
            .unwrap()
            .reverse()
    });

    assert!(retention_amount >= 0. && retention_amount <= 1.);
    chunks.drain((chunks.len() as f32 * retention_amount) as usize..);

    chunks.sort_by(|a, b| a.responce.time.partial_cmp(&b.responce.time).unwrap());

    for mem in long_term_memory {
        println!("ltm: {}, {:?}", mem.summary, mem.emotion)
    }

    for m in chunks {
        for c in m.conditions {
            match c {
                Condition::User(message) => println!("user: {}", message.content),
                Condition::System(message) => println!("system: {}", message.content),
            }
        }
        println!("asis: {}", m.responce.content)
    }

    // todo!()
    Default::default()
}
