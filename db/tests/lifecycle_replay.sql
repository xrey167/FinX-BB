\set ON_ERROR_STOP on

-- Capture the queued key as owner, then perform every mutation as the runtime role.
SELECT outbox_id::text AS ci_outbox_id, ts::text AS ci_outbox_ts
  FROM moda_outbox
 WHERE pseudo_id = 'ci-subject'
   AND generation = 1
\gset

SET ROLE moda_app;

-- Lost responses remain idempotent after both DELETE and a database restart.
DO $$
DECLARE
    v_generation bigint;
    v_duplicate boolean;
    v_rejected boolean := false;
BEGIN
    SELECT receipt.generation, receipt.duplicate
      INTO v_generation, v_duplicate
      FROM moda_enqueue_subject_event(
          'ci-subject', 1, '{"segment":"early-access"}'::jsonb, 'ci-event-1'
      ) AS receipt;
    IF v_generation <> 1 OR NOT v_duplicate THEN
        RAISE EXCEPTION 'post-restart event replay did not return the original receipt';
    END IF;

    SELECT receipt.generation, receipt.duplicate
      INTO v_generation, v_duplicate
      FROM moda_tombstone_subject('ci-subject', 1, 'ci-delete-1') AS receipt;
    IF v_generation <> 2 OR NOT v_duplicate THEN
        RAISE EXCEPTION 'post-restart delete replay did not return the original receipt';
    END IF;

    BEGIN
        PERFORM * FROM moda_enqueue_subject_event(
            'ci-subject', 3, '{"segment":"resurrected"}'::jsonb, 'ci-event-3'
        );
    EXCEPTION WHEN SQLSTATE 'P7101' THEN
        v_rejected := true;
    END;
    IF NOT v_rejected THEN
        RAISE EXCEPTION 'terminal tombstone accepted a later generation';
    END IF;

    v_rejected := false;
    BEGIN
        PERFORM * FROM moda_tombstone_subject('ci-subject', 2, 'ci-delete-fresh');
    EXCEPTION WHEN SQLSTATE 'P7101' THEN
        v_rejected := true;
    END;
    IF NOT v_rejected THEN
        RAISE EXCEPTION 'fresh delete key advanced a terminal tombstone';
    END IF;
END;
$$;

-- The pre-delete event is delivered in separate committed transactions after restart.
SELECT moda_apply_outbox(
    :'ci_outbox_id'::uuid, :'ci_outbox_ts'::timestamptz
);
SELECT moda_apply_outbox(
    :'ci_outbox_id'::uuid, :'ci_outbox_ts'::timestamptz
);

RESET ROLE;

DO $$
BEGIN
    IF (SELECT count(*) FROM moda_subject_events WHERE pseudo_id = 'ci-subject') <> 1
       OR (SELECT count(*) FROM moda_outbox WHERE pseudo_id = 'ci-subject') <> 1 THEN
        RAISE EXCEPTION 'post-restart idempotent replay rewrote event or outbox state';
    END IF;

    IF NOT EXISTS (
        SELECT 1
          FROM moda_subject_lifecycle
         WHERE pseudo_id = 'ci-subject'
           AND generation = 2
           AND tombstoned
    ) OR EXISTS (
        SELECT 1 FROM moda_operational_subjects WHERE pseudo_id = 'ci-subject'
    ) OR NOT EXISTS (
        SELECT 1
          FROM moda_warehouse_subjects
         WHERE pseudo_id = 'ci-subject'
           AND generation_floor = 2
           AND current_generation IS NULL
           AND payload IS NULL
    ) THEN
        RAISE EXCEPTION 'post-restart tombstone or projection state is unsafe';
    END IF;

    IF NOT EXISTS (
        SELECT 1
          FROM moda_outbox
         WHERE pseudo_id = 'ci-subject'
           AND generation = 1
           AND status = 'suppressed'
           AND attempts = 2
    ) THEN
        RAISE EXCEPTION 'delayed event was not suppressed twice after restart';
    END IF;

    IF EXISTS (
        SELECT 1
          FROM moda_idempotency
         WHERE idempotency_key IN ('ci-event-3', 'ci-delete-fresh')
    ) THEN
        RAISE EXCEPTION 'rejected terminal command left idempotency state';
    END IF;
END;
$$;
