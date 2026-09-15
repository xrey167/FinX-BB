BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

DO $role$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'moda_app') THEN
        CREATE ROLE moda_app
            NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT
            NOREPLICATION NOBYPASSRLS;
    ELSIF EXISTS (
        SELECT 1
          FROM pg_catalog.pg_roles
         WHERE rolname = 'moda_app'
           AND (rolcanlogin OR rolsuper OR rolcreatedb OR rolcreaterole
                OR rolreplication OR rolbypassrls)
    ) OR EXISTS (
        SELECT 1
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS member_role
            ON member_role.oid = membership.member
         WHERE member_role.rolname = 'moda_app'
    ) THEN
        RAISE EXCEPTION 'existing moda_app role is not an isolated NOLOGIN role';
    END IF;
END;
$role$;

REVOKE CREATE ON SCHEMA public FROM PUBLIC;

CREATE TABLE moda_idempotency (
    idempotency_key text PRIMARY KEY,
    command_kind text NOT NULL CHECK (command_kind IN ('event', 'delete')),
    body_hash text NOT NULL,
    response_id uuid,
    response_generation bigint NOT NULL CHECK (response_generation >= 0),
    response_ts timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE moda_subject_lifecycle (
    pseudo_id text PRIMARY KEY,
    generation bigint NOT NULL DEFAULT 0 CHECK (generation >= 0),
    tombstoned boolean NOT NULL DEFAULT false,
    tombstoned_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    CHECK (tombstoned = (tombstoned_at IS NOT NULL))
);

CREATE TABLE moda_operational_subjects (
    pseudo_id text PRIMARY KEY REFERENCES moda_subject_lifecycle(pseudo_id),
    generation bigint NOT NULL CHECK (generation > 0),
    payload jsonb NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE moda_subject_events (
    event_id uuid NOT NULL DEFAULT gen_random_uuid(),
    ts timestamptz NOT NULL DEFAULT clock_timestamp(),
    pseudo_id text NOT NULL,
    generation bigint NOT NULL CHECK (generation > 0),
    payload jsonb NOT NULL,
    idempotency_key text NOT NULL,
    body_hash text NOT NULL,
    PRIMARY KEY (event_id, ts)
) PARTITION BY RANGE (ts);

CREATE TABLE moda_subject_events_default
    PARTITION OF moda_subject_events DEFAULT;

CREATE INDEX moda_subject_events_subject_generation_idx
    ON moda_subject_events (pseudo_id, generation DESC, ts DESC);

CREATE TABLE moda_outbox (
    outbox_id uuid NOT NULL DEFAULT gen_random_uuid(),
    ts timestamptz NOT NULL DEFAULT clock_timestamp(),
    event_id uuid NOT NULL,
    event_ts timestamptz NOT NULL,
    pseudo_id text NOT NULL,
    generation bigint NOT NULL CHECK (generation > 0),
    payload jsonb NOT NULL,
    idempotency_key text NOT NULL,
    status text NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'applied', 'suppressed')),
    attempts integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    last_attempt_at timestamptz,
    PRIMARY KEY (outbox_id, ts),
    FOREIGN KEY (event_id, event_ts)
        REFERENCES moda_subject_events(event_id, ts)
) PARTITION BY RANGE (ts);

CREATE TABLE moda_outbox_default
    PARTITION OF moda_outbox DEFAULT;

CREATE INDEX moda_outbox_delivery_idx
    ON moda_outbox (status, ts, outbox_id);

CREATE INDEX moda_outbox_subject_generation_idx
    ON moda_outbox (pseudo_id, generation DESC, ts DESC);

CREATE TABLE moda_warehouse_subjects (
    pseudo_id text PRIMARY KEY,
    generation_floor bigint NOT NULL DEFAULT 0 CHECK (generation_floor >= 0),
    current_generation bigint,
    payload jsonb,
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    CHECK (
        (current_generation IS NULL AND payload IS NULL)
        OR (
            current_generation IS NOT NULL
            AND current_generation > generation_floor
            AND payload IS NOT NULL
        )
    )
);

CREATE OR REPLACE FUNCTION moda_reject_generation_regression()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.generation <> OLD.generation
       AND NEW.generation <> OLD.generation + 1 THEN
        RAISE EXCEPTION 'subject generation must remain % or advance exactly to %, got %',
            OLD.generation, OLD.generation + 1, NEW.generation
            USING ERRCODE = '23514';
    END IF;
    IF OLD.tombstoned
       AND (NOT NEW.tombstoned OR NEW.generation <> OLD.generation) THEN
        RAISE EXCEPTION 'terminally deleted subject % is immutable',
            OLD.pseudo_id
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER moda_subject_generation_monotonic
BEFORE UPDATE ON moda_subject_lifecycle
FOR EACH ROW EXECUTE FUNCTION moda_reject_generation_regression();

CREATE OR REPLACE FUNCTION moda_reject_tombstone_delete()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.tombstoned THEN
        RAISE EXCEPTION 'terminally deleted subject % cannot be removed',
            OLD.pseudo_id
            USING ERRCODE = '23514';
    END IF;
    RETURN OLD;
END;
$$;

CREATE TRIGGER moda_tombstone_row_terminal
BEFORE DELETE ON moda_subject_lifecycle
FOR EACH ROW EXECUTE FUNCTION moda_reject_tombstone_delete();

CREATE OR REPLACE FUNCTION moda_guard_operational_subject()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    v_head moda_subject_lifecycle%ROWTYPE;
BEGIN
    SELECT * INTO v_head
      FROM moda_subject_lifecycle
     WHERE pseudo_id = NEW.pseudo_id
     FOR SHARE;

    IF NOT FOUND OR v_head.tombstoned OR NEW.generation <> v_head.generation THEN
        RAISE EXCEPTION 'operational subject generation is not current for %',
            NEW.pseudo_id
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER moda_operational_subject_current
BEFORE INSERT OR UPDATE ON moda_operational_subjects
FOR EACH ROW EXECUTE FUNCTION moda_guard_operational_subject();

CREATE OR REPLACE FUNCTION moda_reject_warehouse_floor_regression()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.generation_floor < OLD.generation_floor THEN
        RAISE EXCEPTION 'warehouse generation floor cannot decrease from % to %',
            OLD.generation_floor, NEW.generation_floor
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER moda_warehouse_floor_monotonic
BEFORE UPDATE ON moda_warehouse_subjects
FOR EACH ROW EXECUTE FUNCTION moda_reject_warehouse_floor_regression();

CREATE OR REPLACE FUNCTION moda_enqueue_subject_event(
    p_pseudo_id text,
    p_generation bigint,
    p_payload jsonb,
    p_idempotency_key text
)
RETURNS TABLE(
    event_id uuid,
    pseudo_id text,
    generation bigint,
    ts timestamptz,
    duplicate boolean
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    v_body_hash text;
    v_existing moda_idempotency%ROWTYPE;
    v_head moda_subject_lifecycle%ROWTYPE;
    v_event_id uuid := gen_random_uuid();
    v_outbox_id uuid := gen_random_uuid();
    v_ts timestamptz := clock_timestamp();
BEGIN
    IF p_generation <= 0 THEN
        RAISE EXCEPTION 'generation must be positive' USING ERRCODE = '22023';
    END IF;

    v_body_hash := encode(
        digest(
            convert_to(
                jsonb_build_object(
                    'kind', 'event',
                    'pseudoId', p_pseudo_id,
                    'generation', p_generation,
                    'payload', p_payload
                )::text,
                'UTF8'
            ),
            'sha256'
        ),
        'hex'
    );

    PERFORM pg_advisory_xact_lock(hashtextextended(p_idempotency_key, 0));

    SELECT * INTO v_existing
      FROM moda_idempotency
     WHERE idempotency_key = p_idempotency_key;

    IF FOUND THEN
        IF v_existing.command_kind <> 'event' OR v_existing.body_hash <> v_body_hash THEN
            RAISE EXCEPTION 'idempotency key collision: %', p_idempotency_key
                USING ERRCODE = '23505';
        END IF;
        RETURN QUERY
        SELECT v_existing.response_id, p_pseudo_id,
               v_existing.response_generation, v_existing.response_ts, true;
        RETURN;
    END IF;

    INSERT INTO moda_subject_lifecycle (pseudo_id)
    VALUES (p_pseudo_id)
    ON CONFLICT ON CONSTRAINT moda_subject_lifecycle_pkey DO NOTHING;

    SELECT * INTO v_head
      FROM moda_subject_lifecycle AS lifecycle
     WHERE lifecycle.pseudo_id = p_pseudo_id
     FOR UPDATE;

    IF v_head.tombstoned THEN
        RAISE EXCEPTION 'subject % is terminally deleted', p_pseudo_id
            USING ERRCODE = 'P7101';
    END IF;

    IF p_generation <> v_head.generation + 1 THEN
        RAISE EXCEPTION 'generation % must equal next generation % for %',
            p_generation, v_head.generation + 1, p_pseudo_id
            USING ERRCODE = 'P7102';
    END IF;

    UPDATE moda_subject_lifecycle
       SET generation = p_generation,
           tombstoned = false,
           tombstoned_at = NULL,
           updated_at = v_ts
     WHERE moda_subject_lifecycle.pseudo_id = p_pseudo_id;

    INSERT INTO moda_operational_subjects (pseudo_id, generation, payload, updated_at)
    VALUES (p_pseudo_id, p_generation, p_payload, v_ts)
    ON CONFLICT ON CONSTRAINT moda_operational_subjects_pkey DO UPDATE
       SET generation = EXCLUDED.generation,
           payload = EXCLUDED.payload,
           updated_at = EXCLUDED.updated_at;

    INSERT INTO moda_subject_events (
        event_id, ts, pseudo_id, generation, payload, idempotency_key, body_hash
    ) VALUES (
        v_event_id, v_ts, p_pseudo_id, p_generation, p_payload,
        p_idempotency_key, v_body_hash
    );

    INSERT INTO moda_outbox (
        outbox_id, ts, event_id, event_ts, pseudo_id, generation,
        payload, idempotency_key
    ) VALUES (
        v_outbox_id, v_ts, v_event_id, v_ts, p_pseudo_id, p_generation,
        p_payload, p_idempotency_key
    );

    INSERT INTO moda_idempotency (
        idempotency_key, command_kind, body_hash, response_id,
        response_generation, response_ts
    ) VALUES (
        p_idempotency_key, 'event', v_body_hash, v_event_id,
        p_generation, v_ts
    );

    RETURN QUERY SELECT v_event_id, p_pseudo_id, p_generation, v_ts, false;
END;
$$;

CREATE OR REPLACE FUNCTION moda_tombstone_subject(
    p_pseudo_id text,
    p_expected_generation bigint,
    p_idempotency_key text
)
RETURNS TABLE(
    pseudo_id text,
    generation bigint,
    ts timestamptz,
    duplicate boolean
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    v_body_hash text;
    v_existing moda_idempotency%ROWTYPE;
    v_head moda_subject_lifecycle%ROWTYPE;
    v_generation bigint;
    v_ts timestamptz := clock_timestamp();
BEGIN
    v_body_hash := encode(
        digest(
            convert_to(
                'delete:' || p_pseudo_id || ':' || p_expected_generation::text,
                'UTF8'
            ),
            'sha256'
        ),
        'hex'
    );

    PERFORM pg_advisory_xact_lock(hashtextextended(p_idempotency_key, 0));

    SELECT * INTO v_existing
      FROM moda_idempotency
     WHERE idempotency_key = p_idempotency_key;

    IF FOUND THEN
        IF v_existing.command_kind <> 'delete' OR v_existing.body_hash <> v_body_hash THEN
            RAISE EXCEPTION 'idempotency key collision: %', p_idempotency_key
                USING ERRCODE = '23505';
        END IF;
        RETURN QUERY
        SELECT p_pseudo_id, v_existing.response_generation,
               v_existing.response_ts, true;
        RETURN;
    END IF;

    INSERT INTO moda_subject_lifecycle (pseudo_id)
    VALUES (p_pseudo_id)
    ON CONFLICT ON CONSTRAINT moda_subject_lifecycle_pkey DO NOTHING;

    SELECT * INTO v_head
      FROM moda_subject_lifecycle AS lifecycle
     WHERE lifecycle.pseudo_id = p_pseudo_id
     FOR UPDATE;

    IF v_head.tombstoned THEN
        RAISE EXCEPTION 'subject % is terminally deleted', p_pseudo_id
            USING ERRCODE = 'P7101';
    END IF;

    IF p_expected_generation <> v_head.generation THEN
        RAISE EXCEPTION 'expected generation % does not match current generation % for %',
            p_expected_generation, v_head.generation, p_pseudo_id
            USING ERRCODE = 'P7103';
    END IF;

    IF v_head.generation = 9223372036854775807 THEN
        RAISE EXCEPTION 'generation overflow for %', p_pseudo_id
            USING ERRCODE = '22003';
    END IF;

    v_generation := v_head.generation + 1;

    UPDATE moda_subject_lifecycle
       SET generation = v_generation,
           tombstoned = true,
           tombstoned_at = v_ts,
           updated_at = v_ts
     WHERE moda_subject_lifecycle.pseudo_id = p_pseudo_id;

    DELETE FROM moda_operational_subjects AS operational
     WHERE operational.pseudo_id = p_pseudo_id;

    INSERT INTO moda_warehouse_subjects (
        pseudo_id, generation_floor, current_generation, payload, updated_at
    ) VALUES (
        p_pseudo_id, v_generation, NULL, NULL, v_ts
    )
    ON CONFLICT ON CONSTRAINT moda_warehouse_subjects_pkey DO UPDATE
       SET generation_floor = GREATEST(
               moda_warehouse_subjects.generation_floor,
               EXCLUDED.generation_floor
           ),
           current_generation = NULL,
           payload = NULL,
           updated_at = EXCLUDED.updated_at;

    INSERT INTO moda_idempotency (
        idempotency_key, command_kind, body_hash, response_id,
        response_generation, response_ts
    ) VALUES (
        p_idempotency_key, 'delete', v_body_hash, NULL,
        v_generation, v_ts
    );

    RETURN QUERY SELECT p_pseudo_id, v_generation, v_ts, false;
END;
$$;

CREATE OR REPLACE FUNCTION moda_apply_outbox(
    p_outbox_id uuid,
    p_outbox_ts timestamptz
)
RETURNS text
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    v_outbox moda_outbox%ROWTYPE;
    v_warehouse moda_warehouse_subjects%ROWTYPE;
    v_status text;
BEGIN
    SELECT * INTO v_outbox
      FROM moda_outbox
     WHERE outbox_id = p_outbox_id AND ts = p_outbox_ts
     FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'outbox record not found' USING ERRCODE = 'P0002';
    END IF;

    INSERT INTO moda_warehouse_subjects (pseudo_id)
    VALUES (v_outbox.pseudo_id)
    ON CONFLICT ON CONSTRAINT moda_warehouse_subjects_pkey DO NOTHING;

    SELECT * INTO v_warehouse
      FROM moda_warehouse_subjects
     WHERE pseudo_id = v_outbox.pseudo_id
     FOR UPDATE;

    IF v_outbox.generation <= v_warehouse.generation_floor THEN
        v_status := 'suppressed';
    ELSIF v_warehouse.current_generation = v_outbox.generation THEN
        v_status := 'applied';
    ELSIF v_warehouse.current_generation IS NOT NULL
          AND v_outbox.generation < v_warehouse.current_generation THEN
        v_status := 'suppressed';
    ELSE
        UPDATE moda_warehouse_subjects
           SET current_generation = v_outbox.generation,
               payload = v_outbox.payload,
               updated_at = clock_timestamp()
         WHERE pseudo_id = v_outbox.pseudo_id;
        v_status := 'applied';
    END IF;

    UPDATE moda_outbox
       SET status = v_status,
           attempts = attempts + 1,
           last_attempt_at = clock_timestamp()
     WHERE outbox_id = p_outbox_id AND ts = p_outbox_ts;

    RETURN v_status;
END;
$$;

REVOKE INSERT, UPDATE, DELETE, TRUNCATE
ON TABLE
    moda_idempotency,
    moda_subject_lifecycle,
    moda_operational_subjects,
    moda_subject_events,
    moda_subject_events_default,
    moda_outbox,
    moda_outbox_default,
    moda_warehouse_subjects
FROM PUBLIC;

REVOKE ALL PRIVILEGES
ON TABLE
    moda_idempotency,
    moda_subject_lifecycle,
    moda_operational_subjects,
    moda_subject_events,
    moda_subject_events_default,
    moda_outbox,
    moda_outbox_default,
    moda_warehouse_subjects
FROM moda_app;

REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public FROM moda_app;

REVOKE EXECUTE ON FUNCTION public.moda_reject_generation_regression() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION public.moda_reject_tombstone_delete() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION public.moda_guard_operational_subject() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION public.moda_reject_warehouse_floor_regression() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION public.moda_enqueue_subject_event(text, bigint, jsonb, text) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION public.moda_tombstone_subject(text, bigint, text) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION public.moda_apply_outbox(uuid, timestamptz) FROM PUBLIC;

GRANT USAGE ON SCHEMA public TO moda_app;
GRANT EXECUTE ON FUNCTION public.moda_enqueue_subject_event(text, bigint, jsonb, text) TO moda_app;
GRANT EXECUTE ON FUNCTION public.moda_tombstone_subject(text, bigint, text) TO moda_app;
GRANT EXECUTE ON FUNCTION public.moda_apply_outbox(uuid, timestamptz) TO moda_app;

COMMIT;
