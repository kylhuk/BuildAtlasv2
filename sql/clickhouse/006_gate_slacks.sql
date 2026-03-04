-- Add gate_slacks column to scenario_metrics table
ALTER TABLE scenario_metrics
ADD COLUMN IF NOT EXISTS gate_slacks Nested(
    resist_fire_slack Float64,
    resist_cold_slack Float64,
    resist_lightning_slack Float64,
    resist_chaos_slack Float64,
    max_hit_slack Float64,
    min_gate_slack Float64,
    num_gate_violations UInt8
);
