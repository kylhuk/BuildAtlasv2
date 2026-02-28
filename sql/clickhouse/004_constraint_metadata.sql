ALTER TABLE builds
ADD COLUMN IF NOT EXISTS constraint_status Nullable(String),
ADD COLUMN IF NOT EXISTS constraint_reason_code Nullable(String),
ADD COLUMN IF NOT EXISTS violated_constraints Array(String) DEFAULT [],
ADD COLUMN IF NOT EXISTS constraint_checked_at Nullable(DateTime64(3, 'UTC'));
