-- wrp/server/runtime/store/sql/sqlite/schema.sql
-- runs
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  workflow_name TEXT NOT NULL,
  thread_id TEXT NULL,
  created_at TEXT NOT NULL,
  state TEXT NOT NULL,
  message_count INTEGER NOT NULL DEFAULT 0,
  channel_counts_json TEXT NOT NULL DEFAULT '{}',
  outcome TEXT NULL,
  error_text TEXT NULL,
  run_output_blob BLOB NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_thread ON runs(workflow_name, thread_id, created_at);

-- global counters
CREATE TABLE IF NOT EXISTS counters (
  key TEXT PRIMARY KEY,
  value INTEGER NOT NULL
);

-- conversation items
CREATE TABLE IF NOT EXISTS conversation_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  idx INTEGER NOT NULL,
  ts TEXT NOT NULL,
  sort_ts TEXT NOT NULL,
  channel TEXT NOT NULL,
  payload_blob BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conv_run_ts ON conversation_items(run_id, ts);
CREATE INDEX IF NOT EXISTS idx_conv_run_ch_ts ON conversation_items(run_id, channel, ts);

-- telemetry
CREATE TABLE IF NOT EXISTS telemetry_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  ts TEXT NOT NULL,
  kind TEXT NOT NULL,
  payload_blob BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tlm_run_ts ON telemetry_events(run_id, ts);
CREATE INDEX IF NOT EXISTS idx_tlm_run_kind_ts ON telemetry_events(run_id, kind, ts);

-- span payloads
CREATE TABLE IF NOT EXISTS span_payloads (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  span_id TEXT NOT NULL,
  envelope_blob BLOB NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(run_id, span_id)
);

-- workflow settings
CREATE TABLE IF NOT EXISTS workflow_settings (
  workflow_name TEXT PRIMARY KEY,
  values_json TEXT NOT NULL,
  overridden INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL
);