-- EP-V3-02 Slice A: additive migration for build cost summarization
CREATE TABLE IF NOT EXISTS build_costs (
    build_id String,
    ruleset_id String,
    price_snapshot_id String,
    total_cost_chaos Float64,
    unknown_cost_count UInt32,
    slot_costs_json_path String,
    gem_costs_json_path String,
    calculated_at DateTime64(3, 'UTC')
) ENGINE = MergeTree()
ORDER BY (build_id, price_snapshot_id, ruleset_id, calculated_at);
