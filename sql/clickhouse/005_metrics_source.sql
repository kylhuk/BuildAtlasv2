ALTER TABLE scenario_metrics
ADD COLUMN IF NOT EXISTS metrics_source String DEFAULT 'pob';
