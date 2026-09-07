# FinX-Moda lifecycle API (vertical slice)

This is the first backend slice for resurrection-safe subject events. It keeps
the domain rules behind a `LifecycleRepository`: PostgreSQL 16 is the intended
durable implementation, while the included in-memory adapter makes the API and
tests runnable without Docker.

## Run and test

Node.js 22 or newer is required.

```bash
cd apps/moda-api
npm ci
npm test
npm run build
npm run dev
```

The development server binds to `127.0.0.1:3001` by default. Set `HOST`
explicitly only when remote access is intended. A non-loopback bind is refused
unless the deliberately conspicuous demo opt-in is present:

```bash
HOST=0.0.0.0 INSECURE_REMOTE_DEMO=1 npm run dev
```

This flag does not add authentication or make remote exposure safe; it only
prevents an accidental bind. Keep the API behind loopback unless an external
authenticated gateway is in place.

```bash
curl http://localhost:3001/health

curl -X POST http://localhost:3001/v1/events \
  -H 'content-type: application/json' \
  -d '{"pseudoId":"demo-1","generation":"1","payload":{"look":"minimal"},"idempotencyKey":"event-1"}'

curl -X POST http://localhost:3001/v1/outbox/deliver \
  -H 'content-type: application/json' \
  -d '{}'

curl -X POST http://localhost:3001/v1/subjects/demo-1/delete \
  -H 'content-type: application/json' \
  -d '{"expectedGeneration":"1","idempotencyKey":"delete-1"}'

curl http://localhost:3001/v1/state
```

To inspect the PostgreSQL 16 target schema and stored transaction functions,
run the following from the repository root. If you are continuing in the shell
used above, return there first:

```bash
cd "$(git rev-parse --show-toplevel)"
docker compose -f docker-compose.lifecycle.yml up -d
docker compose -f docker-compose.lifecycle.yml exec lifecycle-postgres \
  psql -U finx -d finx -c '\\dt moda_*'
```

The Compose `finx` identity owns the local database and is for migrations and
inspection only. The current API does not connect to PostgreSQL. A deployment
that adds the PostgreSQL adapter should provision its login separately and make
it a member of the migration-created, non-login capability role:

```sql
CREATE ROLE moda_api_login LOGIN PASSWORD '<from-secret-manager>';
GRANT moda_app TO moda_api_login;
```

Connect the service as `moda_api_login`, never as the database or migration
owner. `moda_app` has no direct table or sequence privileges. It can execute
only `moda_enqueue_subject_event`, `moda_tombstone_subject`, and
`moda_apply_outbox`; those functions run as their migration owner with a pinned
`pg_catalog, public, pg_temp` search path. `CREATE` on `public` is revoked from
`PUBLIC` to keep that path trusted.

Still from the repository root, remove the local database and its volume with:

```bash
docker compose -f docker-compose.lifecycle.yml down -v
```

## Lifecycle contract

- An accepted event advances a subject by exactly one generation (a new subject
  starts at generation 1) and
  creates exactly one outbox record for an idempotency key.
- Reusing a key with the same command returns the original result. Reusing it
  with a different body returns HTTP 409. Event and delete receipts contain no
  payload, including when replayed after deletion or restart.
- Deletion requires the caller's `expectedGeneration`. A mismatch returns HTTP
  409 without changing lifecycle, projections, outbox, or idempotency state.
  A match advances the lifecycle head by one generation, removes the
  operational projection, records a tombstone, and raises the warehouse
  generation floor in the same repository operation.
- Delivery applies an event only when its generation is strictly newer than the
  warehouse floor and current projection. Delayed pre-delete events are
  suppressed, including after snapshot restore and replay.
- A tombstoned `pseudoId` is terminal and cannot be reused at any later
  generation. Replaying the original delete key returns its receipt; a fresh
  delete key receives a terminal-subject error and cannot advance generation.
  A new incarnation therefore requires a new `pseudoId`.
- All HTTP generation inputs and outputs are canonical unsigned decimal strings.
  Internally the adapter uses `bigint`, avoiding JavaScript number precision
  loss and matching PostgreSQL's signed `bigint` upper bound.

## Scope limits

The running API currently uses process-local memory. Restart persistence is
demonstrated by snapshot/restore in tests, not wired to PostgreSQL. The migration
defines the intended PostgreSQL tables and atomic functions, but a TypeScript
PostgreSQL adapter, authentication, tenancy, retries, partition rotation,
retention, and production observability are not included.

The migration cannot define the final login or its secret for an unknown
deployment. Database owners can bypass ordinary grants, so owner credentials
must remain outside the application runtime. If separate API and outbox-worker
roles are later needed, split the three function grants between narrower
non-login roles rather than adding table privileges.

This demo does **not** prove distributed deletion. It only demonstrates the
generation-floor rule for the included operational and warehouse projections;
it does not purge caches, object stores, backups, model weights, or external
systems.

Events still infer the required next generation from their supplied
`generation`; a separate event-side `expectedGeneration` CAS field is a
follow-up slice. Deletes already require that explicit precondition. This MVP
rejects stale, duplicate-with-a-new-key, and gap generations by requiring the
exact next value.
