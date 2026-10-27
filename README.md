# Associative Memory AI Companion

A console AI companion with emotions, a sense of time and a memory that fades like a person's.
See [doc/spec.md](doc/spec.md) for the specification.

## Usage

```sh
export OPENAI_API_KEY=sk-...
cargo run --release                      # chat with the "default" companion
cargo run --release -- --profile alice   # another companion
cargo run --release -- profiles          # list companions
cargo run --release -- --help
```

On first start of a profile the companion chooses its own persona (or you write it yourself).
Inside the chat type `/help` for commands (`/emotions`, `/persona`, `/memories [query]`,
`/facts`, `/compact`, `/debug`, `/exit`).

## Data

Default data dir: `~/.associative_memory_ai_companion/` (override with `--data-dir` or
`COMPANION_DATA_DIR`).

```
config.json                   model, model catalog the companion knows about,
                              memory budgets, thresholds (created with defaults)
profiles/<name>/
  persona.json                who the companion is — edit freely, they will notice
  persona_meta.json           who wrote each persona field + last seen snapshot
  state.json                  user name, current emotions, last interaction time,
                              the model it ran on last (to notice upgrades/downgrades)
  short_term.json             recent raw messages (with emotions and inner thoughts)
  long_term.json              episodic memories and core facts
  embeddings.json             vectors for associative recall
```

## How memory works

1. Messages are kept verbatim in short-term memory until it exceeds `short_term_max_tokens`.
2. The oldest messages are then summarised into episodic memories (with emotions and
   importance) and new core facts.
3. Each memory has a retention score = emotional intensity + importance + recency
   (half-life `memory_half_life_days`, reset whenever the memory is recalled).
4. When there are more than `max_memories`, the weakest ones are merged into a single vaguer
   memory. Core facts are never dropped, only condensed.
5. Each turn, the memories most similar to the conversation (embeddings × retention) are
   recalled into the prompt, along with core facts and the latest memories.
