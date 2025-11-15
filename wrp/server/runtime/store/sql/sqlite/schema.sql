# wrp/server/runtime/store/sql/sqlite/schema.sql

-- system sessions
CREATE TABLE IF NOT EXISTS system_sessions (
  system_session_id TEXT PRIMARY KEY,
  name TEXT NULL,
  created_at TEXT NOT NULL
);

-- runs
CREATE TABLE IF NOT EXISTS runs (
  system_session_id TEXT NOT NULL REFERENCES system_sessions(system_session_id) ON DELETE CASCADE,
  run_id TEXT NOT NULL,
  workflow_name TEXT NOT NULL,
  thread_id TEXT NULL,
  created_at TEXT NOT NULL,
  state TEXT NOT NULL,
  outcome TEXT NULL,
  error_text TEXT NULL,
  run_output_blob BLOB NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (system_session_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_runs_thread ON runs(system_session_id, workflow_name, thread_id, created_at);

-- per-session counters
CREATE TABLE IF NOT EXISTS counters (
  system_session_id TEXT NOT NULL REFERENCES system_sessions(system_session_id) ON DELETE CASCADE,
  key TEXT NOT NULL,
  value INTEGER NOT NULL,
  PRIMARY KEY (system_session_id, key)
);

-- conversation items
CREATE TABLE IF NOT EXISTS conversation_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  system_session_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  idx INTEGER NOT NULL,
  ts TEXT NOT NULL,
  sort_ts TEXT NOT NULL,
  channel TEXT NOT NULL,
  payload_blob BLOB NOT NULL,
  FOREIGN KEY (system_session_id, run_id)
    REFERENCES runs(system_session_id, run_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_conv_run_ts ON conversation_items(system_session_id, run_id, ts);
CREATE INDEX IF NOT EXISTS idx_conv_run_ch_ts ON conversation_items(system_session_id, run_id, channel, ts);

-- conversation channel metadata (per run+channel)
CREATE TABLE IF NOT EXISTS conversation_channels (
  system_session_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  channel TEXT NOT NULL,
  name TEXT NULL,
  description TEXT NULL,
  items_count INTEGER NOT NULL DEFAULT 0,
  last_ts TEXT NULL,
  PRIMARY KEY (system_session_id, run_id, channel),
  FOREIGN KEY (system_session_id, run_id)
    REFERENCES runs(system_session_id, run_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_conv_ch_last_ts ON conversation_channels(system_session_id, run_id, last_ts);

-- telemetry
CREATE TABLE IF NOT EXISTS telemetry_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  system_session_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  ts TEXT NOT NULL,
  kind TEXT NOT NULL,
  payload_blob BLOB NOT NULL,
  FOREIGN KEY (system_session_id, run_id)
    REFERENCES runs(system_session_id, run_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_tlm_run_ts ON telemetry_events(system_session_id, run_id, ts);
CREATE INDEX IF NOT EXISTS idx_tlm_run_kind_ts ON telemetry_events(system_session_id, run_id, kind, ts);

-- span payloads
CREATE TABLE IF NOT EXISTS span_payloads (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  system_session_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  span_id TEXT NOT NULL,
  envelope_blob BLOB NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(system_session_id, run_id, span_id),
  FOREIGN KEY (system_session_id, run_id)
    REFERENCES runs(system_session_id, run_id) ON DELETE CASCADE
);

-- workflow settings
CREATE TABLE IF NOT EXISTS workflow_settings (
  workflow_name TEXT PRIMARY KEY,
  values_json TEXT NOT NULL,
  overridden INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL
);

-- provider settings (encrypted values)
CREATE TABLE IF NOT EXISTS provider_settings (
  provider_name TEXT PRIMARY KEY,
  values_blob BLOB NOT NULL,
  overridden INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL
);

-- agent settings (plaintext JSON; no secrets)
CREATE TABLE IF NOT EXISTS agent_settings (
  agent_name TEXT PRIMARY KEY,
  values_json TEXT NOT NULL,
  overridden INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL
);