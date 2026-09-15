\set ON_ERROR_STOP on

DO $$
BEGIN
    IF current_setting('server_version_num')::integer < 160000
       OR current_setting('server_version_num')::integer >= 170000 THEN
        RAISE EXCEPTION 'expected PostgreSQL 16, got %', current_setting('server_version');
    END IF;
END;
$$;

-- Commit the authoritative state through the function-only runtime role.
BEGIN;
SET LOCAL ROLE moda_app;

DO $$
DECLARE
    v_rejected boolean := false;
BEGIN
    BEGIN
        INSERT INTO moda_subject_lifecycle (pseudo_id) VALUES ('ci-direct-dml');
    EXCEPTION WHEN insufficient_privilege THEN
        v_rejected := true;
    END;
    IF NOT v_rejected THEN
        RAISE EXCEPTION 'moda_app unexpectedly received direct table DML';
    END IF;
END;
$$;

SELECT * FROM moda_enqueue_subject_event(
    'ci-subject', 1, '{"segment":"early-access"}'::jsonb, 'ci-event-1'
);
SELECT * FROM moda_tombstone_subject('ci-subject', 1, 'ci-delete-1');

-- A delete authorized against generation 1 must not delete generation 2.
SELECT * FROM moda_enqueue_subject_event(
    'ci-cas', 1, '{"version":1}'::jsonb, 'ci-cas-event-1'
);
SELECT * FROM moda_enqueue_subject_event(
    'ci-cas', 2, '{"version":2}'::jsonb, 'ci-cas-event-2'
);

DO $$
DECLARE
    v_rejected boolean := false;
BEGIN
    BEGIN
        PERFORM * FROM moda_tombstone_subject('ci-cas', 1, 'ci-cas-stale-delete');
    EXCEPTION WHEN SQLSTATE 'P7103' THEN
        v_rejected := true;
    END;
    IF NOT v_rejected THEN
        RAISE EXCEPTION 'stale delete CAS was accepted';
    END IF;

    v_rejected := false;
    BEGIN
        PERFORM * FROM moda_enqueue_subject_event(
            'ci-gap', 2, '{"value":"gap"}'::jsonb, 'ci-gap-event'
        );
    EXCEPTION WHEN SQLSTATE 'P7102' THEN
        v_rejected := true;
    END;
    IF NOT v_rejected THEN
        RAISE EXCEPTION 'generation gap was accepted';
    END IF;
END;
$$;

COMMIT;

-- These assertions run as the migration owner after the runtime transaction commits.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM moda_subject_lifecycle
         WHERE pseudo_id = 'ci-subject'
           AND generation = 2
           AND tombstoned
    ) OR EXISTS (
        SELECT 1 FROM moda_operational_subjects WHERE pseudo_id = 'ci-subject'
    ) THEN
        RAISE EXCEPTION 'committed tombstone or operational purge is unsafe';
    END IF;

    IF NOT EXISTS (
        SELECT 1
          FROM moda_warehouse_subjects
         WHERE pseudo_id = 'ci-subject'
           AND generation_floor = 2
           AND current_generation IS NULL
           AND payload IS NULL
    ) OR NOT EXISTS (
        SELECT 1
          FROM moda_outbox
         WHERE pseudo_id = 'ci-subject'
           AND generation = 1
           AND status = 'queued'
           AND attempts = 0
    ) THEN
        RAISE EXCEPTION 'pre-restart warehouse floor or delayed outbox is unsafe';
    END IF;

    IF NOT EXISTS (
        SELECT 1
          FROM moda_subject_lifecycle
         WHERE pseudo_id = 'ci-cas'
           AND generation = 2
           AND NOT tombstoned
    ) OR NOT EXISTS (
        SELECT 1
          FROM moda_operational_subjects
         WHERE pseudo_id = 'ci-cas'
           AND generation = 2
           AND payload = '{"version":2}'::jsonb
    ) OR EXISTS (
        SELECT 1
          FROM moda_idempotency
         WHERE idempotency_key IN ('ci-cas-stale-delete', 'ci-gap-event')
    ) THEN
        RAISE EXCEPTION 'rejected CAS or gap changed committed state';
    END IF;
END;
$$;
