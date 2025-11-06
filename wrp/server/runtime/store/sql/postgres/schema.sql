-- wrp/server/runtime/store/sql/postgres/schema.sql
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'runs') THEN

        CREATE TABLE runs (
          run_id TEXT PRIMARY KEY,
          workflow_name TEXT NOT NULL,
          thread_id TEXT NULL,
          created_at TEXT NOT NULL,
          state TEXT NOT NULL,
          message_count INTEGER NOT NULL DEFAULT 0,
          channel_counts_json TEXT NOT NULL DEFAULT '{}',
          outcome TEXT NULL,
          error_text TEXT NULL,
          run_output_blob BYTEA NULL,
          updated_at TEXT NOT NULL
        );
        CREATE INDEX idx_runs_thread ON runs(workflow_name, thread_id, created_at);

        CREATE TABLE counters (
          key TEXT PRIMARY KEY,
          value INTEGER NOT NULL
        );

        CREATE TABLE conversation_items (
          id BIGSERIAL PRIMARY KEY,
          run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
          idx INTEGER NOT NULL,
          ts TEXT NOT NULL,
          sort_ts TEXT NOT NULL,
          channel TEXT NOT NULL,
          payload_blob BYTEA NOT NULL
        );
        CREATE INDEX idx_conv_run_ts ON conversation_items(run_id, ts);
        CREATE INDEX idx_conv_run_ch_ts ON conversation_items(run_id, channel, ts);

        CREATE TABLE telemetry_events (
          id BIGSERIAL PRIMARY KEY,
          run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
          ts TEXT NOT NULL,
          kind TEXT NOT NULL,
          payload_blob BYTEA NOT NULL
        );
        CREATE INDEX idx_tlm_run_ts ON telemetry_events(run_id, ts);
        CREATE INDEX idx_tlm_run_kind_ts ON telemetry_events(run_id, kind, ts);

        CREATE TABLE span_payloads (
          id BIGSERIAL PRIMARY KEY,
          run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
          span_id TEXT NOT NULL,
          envelope_blob BYTEA NOT NULL,
          updated_at TEXT NOT NULL,
          CONSTRAINT uq_span UNIQUE(run_id, span_id)
        );

        CREATE TABLE workflow_settings (
          workflow_name TEXT PRIMARY KEY,
          values_json TEXT NOT NULL,
          overridden INTEGER NOT NULL DEFAULT 0,
          updated_at TEXT NOT NULL
        );

    END IF;
END$$;