CREATE TABLE IF NOT EXISTS builds (
    build_id String,
    created_at DateTime64(3, 'UTC'),
    ruleset_id String,
    profile_id String,
    class String,
    ascendancy String,
    main_skill String,
    damage_type String,
    defence_type String,
    complexity_bucket String,
    pob_xml_path String,
    pob_code_path String,
    genome_path String,
    tags Array(String),
    status String,
    is_stale UInt8 DEFAULT 0
) ENGINE = MergeTree()
ORDER BY (profile_id, ruleset_id, created_at);

CREATE TABLE IF NOT EXISTS scenario_metrics (
    build_id String,
    ruleset_id String,
    scenario_id String,
    gate_pass UInt8,
    gate_fail_reasons Array(String),
    pob_warnings Array(String),
    evaluated_at DateTime64(3, 'UTC'),
    full_dps Float64,
    max_hit Float64,
    armour Float64,
    evasion Float64,
    life Float64,
    mana Float64,
    utility_score Float64,
    metrics_source String DEFAULT 'pob'
) ENGINE = MergeTree()
ORDER BY (scenario_id, ruleset_id, gate_pass, evaluated_at);
