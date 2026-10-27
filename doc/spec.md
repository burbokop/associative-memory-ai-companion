# Specification

## Project description 
AI agent (companion) with emotions that should behave as a real person, that you could be friend with and develop any kind of relationship.

## Features
### Phase 1
- Must use OpenAI API. 
- Should be a console application
- Current emotion should be displayed in chat. 
- It should have memory that it compacted authomatically over time in the way that memories with higher emotions have higher priority to stay as well as more recent memories. 
- It should be able to track time that has passed during conversations and notice when for example too much time has passed and ask "Why did you not talk to me for so long"
- Memory should be stored in json file (files) on file system (default dir ~/.associative_memory_ai_companion/) with ability to specify another dir in cli flags

#### Phase 1 – clarifications
- **Implementation:** rewrite from scratch (previous prototype is discarded).
- **API:** OpenAI Responses API used statelessly (`store: false`) — the app itself decides what the model sees. Structured outputs (JSON schema) are used for replies, emotions and memory operations. Model, embedding model and API base URL are configurable. The companion produces private "inner thoughts" with every reply, which are stored and feed memory consolidation.
- **Associative recall:** each long-term memory has an embedding. Every turn, the memories most similar to the current conversation (weighted by their retention score) are recalled into the prompt together with core facts and the most recent memories. Recalling a memory refreshes it, so frequently recalled memories survive longer.
- **Profiles:** several companions are supported. Each profile lives in its own subdirectory of the data dir (`<data-dir>/profiles/<name>/`) and is selected with a CLI flag.
- **Persona:**
  - Stored in a user-editable file inside the profile directory.
  - On the first start of a new profile the companion creates its own persona via the OpenAI API (picks name, personality, background, etc.). The user may instead provide one.
  - Authorship (companion or user) is tracked per field and updated when changes are detected.
  - The app keeps a snapshot of the last-known persona. On startup the persona file is compared with the snapshot; if the user changed it, the companion is told what changed and may react to it (e.g. "Why did you change this about me?").
- **Self-awareness of the model:** the companion knows it is a digital person whose mind runs on an AI model. It knows which model and reasoning effort it runs on, since when, and which other models exist (a user-editable catalog with descriptions/prices in `config.json`), so it may have wishes about it ("why am I on the cheapest model? upgrade me"). The last used model is recorded per profile; when it changes the companion is told what changed and reacts, the same way as for persona changes.
- **Model per profile:** each companion can run on its own model and reasoning effort (`settings.json` in the profile dir); the global `config.json` values are the default. They can be switched during the chat with `/model <name>` and `/effort <level>` — saved for that companion by default, or for the current run only with `--temp` (the `--model` CLI flag is also for one run only). The companion reacts to the switch right away; if the new model fails, the switch is rolled back.
- **Relationship:** the companion keeps an explicit view of its relationship with the user (`relationship.json`, user-editable):
  - a free-text label in its own words (e.g. "new friends", "something more?"), changed only on reflection and only when the relationship genuinely changed; label changes are kept as history;
  - dimensions 0–100 (trust, closeness, affection, respect, admiration, attraction, playfulness, longing, tension, jealousy; the user may add/remove dimensions), updated after every message in small steps (limited by `relationship_max_step`);
  - a private opinion of the user and wishes for the relationship, rewritten on reflection (when short-term memory is consolidated).
  - Display: label and dimensions are shown in the chat header and via `/relationship`; the private opinion and wishes are only shown in debug mode. The companion knows the user can see the label and numbers, and is told when the user opened `/relationship` (and whether they saw the private parts).
  - Edits by the user are detected (also while the chat is open) and the companion reacts, like for persona edits; authorship is tracked per field.
- **Emotions:** the companion can feel several emotions at once, each with an intensity 0–100 (e.g. "joy 60, anxiety 20"). Emotions persist between turns as a mood and decay towards a baseline over time.
- **Memory compaction:**
  - Short-term memory: raw recent messages up to a token budget.
  - On overflow the oldest messages are summarised into episodic long-term memories (summary, emotions, intensity, importance, timestamps).
  - Retention score combines emotional intensity, importance and recency (exponential decay).
  - Low-score memories are not deleted but merged into vaguer, more general summaries.
  - A small set of core facts (user's name, key relationship facts) is never forgotten.
- **Time tracking:** every message is timestamped and the model sees the current time and the gap since the last message. After a long absence the companion speaks first when the user comes back. Reacting to silence while the app is open is deferred to Phase 3.

### Phase 2
- It should have an ability to run bash commands withing confined docker image without access to internet or host file system. How sould this work: user can run app with `--container [dir]` parameter. In that case application runs docker command (Dockerfile is embedded into the app's binary). The binary is passed to container and run inside with parameter `--bash-access [some connection id]` (probably some safeguard should be added so that it will be impossible to run with this flag on non-docker host). All logic is performed in outer instance of the app. when agent wants to run bash command the app sends message to the instance inside docker, it runs the command and sends exit code, termination signal, stdout, stderr back. From inside of the container there should not be a way to access outer world (no internet connection). It can only access its internal filesystem that is itself stored in `dir` passed to `--container` parameter. It sohuld not be able to access filesystem outside of this folder. the whole system should be completely isolated by linux means, so that there it couldnt be broken by any programe that agent could write inside. 
- The container should be debian 14, have gcc, g++, clang, clang++, rust, python installed additionally some rust packages (like serde, serde_json, rand, tokio) preinstalled since it will be impossible to install them after container is started.

### Phase 3
- It should be able to run as background task make OpenAI requests periodically with some interval to emulate personal time outside of the chat.
### Phase 4
- Voise-over ability
- Customizable visual avatar that works in sync with Voise-over and shows current emotions.
