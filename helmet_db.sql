DROP DATABASE IF EXISTS helmet_db;
CREATE DATABASE helmet_db;
\connect helmet_db;

CREATE TYPE status AS ENUM ('pending','confirmed','rejected');

CREATE TABLE cameras (
    camera_id INT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    location VARCHAR(200),
    ip_address inet,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO cameras (camera_id, name, location, ip_address)
VALUES
    (0,  'Camera 0',  '', NULL),
    (1,  'Camera 1',  '', NULL),
    (2,  'Camera 2',  '', NULL),
    (3,  'Camera 3',  '', NULL),
    (4,  'Camera 4',  '', NULL),
    (5,  'Camera 5',  '', NULL),
    (6,  'Camera 6',  '', NULL),
    (7,  'Camera 7',  '', NULL),
    (8,  'Camera 8',  '', NULL),
    (9,  'Camera 9',  '', NULL),
    (10, 'Camera 10', '', NULL),
    (11, 'Camera 11', '', NULL),
    (12, 'Camera 12', '', NULL),
    (13, 'Camera 13', '', NULL),
    (14, 'Camera 14', '', NULL),
    (15, 'Camera 15', '', NULL)
ON CONFLICT (camera_id) DO NOTHING;

CREATE TABLE admins (
    admin_id SERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(200) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE detections (
    detection_id BIGSERIAL PRIMARY KEY,
    camera_id INT NOT NULL,
    detection_time TIMESTAMPTZ NOT NULL,
    image BYTEA NOT NULL,
    status status NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_detections_camera
        FOREIGN KEY (camera_id) REFERENCES cameras (camera_id)
);

CREATE INDEX idx_detections_status_time
    ON detections (status, detection_time DESC);

CREATE INDEX idx_detections_camera_time
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
