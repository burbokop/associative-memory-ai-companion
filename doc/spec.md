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
- **Self-awareness of the model:** the companion knows it is a digital person whose mind runs on an AI model. It knows which model and reasoning effort it runs on, since when, and which other models exist (a user-editable catalog with descriptions/prices in `config.json`), so it may have wishes about it ("why am I on the cheapest model? upgrade me"). The last used model is recorded per profile; when it changes (config or `--model`) the companion is told what changed and reacts on the next start, the same way as for persona changes.
- **Emotions:** the companion can feel several emotions at once, each with an intensity 0–100 (e.g. "joy 60, anxiety 20"). Emotions persist between turns as a mood and decay towards a baseline over time.
- **Memory compaction:**
  - Short-term memory: raw recent messages up to a token budget.
  - On overflow the oldest messages are summarised into episodic long-term memories (summary, emotions, intensity, importance, timestamps).
  - Retention score combines emotional intensity, importance and recency (exponential decay).
  - Low-score memories are not deleted but merged into vaguer, more general summaries.
  - A small set of core facts (user's name, key relationship facts) is never forgotten.
- **Time tracking:** every message is timestamped and the model sees the current time and the gap since the last message. After a long absence the companion speaks first when the user comes back. Reacting to silence while the app is open is deferred to Phase 3.

### Phase 2
- It should have an ability to run bash commands withing confined docker image without access to internet or host file system.
### Phase 3
- It should be able to run as background task make OpenAI requests periodically with some interval to emulate personal time outside of the chat.
### Phase 4
- Voise-over ability
- Customizable visual avatar that works in sync with Voise-over and shows current emotions.
