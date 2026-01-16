DROP DATABASE IF EXISTS helmet_db;
CREATE DATABASE helmet_db;
\connect helmet_db;

DO $$
BEGIN
    CREATE TYPE status AS ENUM ('pending','confirmed','rejected');
EXCEPTION
    WHEN duplicate_object THEN NULL;
END$$;

CREATE TABLE IF NOT EXISTS detections (
    detection_id BIGSERIAL PRIMARY KEY,
    camera_id INT NOT NULL,
    detection_time TIMESTAMPTZ NOT NULL,
    image BYTEA NOT NULL,
    status status NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_detections_status_time
    ON detections (status, detection_time DESC);

CREATE INDEX IF NOT EXISTS idx_detections_camera_time
    ON detections (camera_id, detection_time DESC);

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at := NOW();
    RETURN NEW;
END$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_set_updated_at ON detections;
CREATE TRIGGER trg_set_updated_at
BEFORE UPDATE ON detections
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE OR REPLACE FUNCTION delete_outdated_rejected()
RETURNS void AS $$
BEGIN
    DELETE FROM detections WHERE status = 'rejected';
END$$ LANGUAGE plpgsql;
