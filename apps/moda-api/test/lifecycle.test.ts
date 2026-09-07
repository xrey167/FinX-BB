import { describe, expect, it } from "vitest";

import { buildApp } from "../src/app.js";
import {
  GenerationConflictError,
  GenerationSequenceError,
  IdempotencyConflictError,
  InMemoryLifecycleRepository,
  SubjectDeletedError,
} from "../src/in-memory-repository.js";
import type { LifecycleRepository } from "../src/repository.js";

describe("subject lifecycle", () => {
  it("cannot resurrect a deleted subject through delayed delivery or restart replay", async () => {
    const repository = new InMemoryLifecycleRepository();

    const delayedEvent = {
      pseudoId: "subject-7",
      generation: 1n,
      payload: { segment: "early-access" },
      idempotencyKey: "event-1",
    } as const;
    const acceptedEvent = repository.enqueueEvent(delayedEvent);
    const deleted = repository.deleteSubject({
      pseudoId: "subject-7",
      expectedGeneration: 1n,
      idempotencyKey: "delete-1",
    });

    expect(deleted.receipt.generation).toBe(2n);
    expect(repository.getState().operationalSubjects).toEqual([]);

    const replayAfterDelete = repository.enqueueEvent(delayedEvent);
    expect(replayAfterDelete.duplicate).toBe(true);
    expect(replayAfterDelete.receipt).toEqual(acceptedEvent.receipt);
    expect(replayAfterDelete.receipt).not.toHaveProperty("payload");
    expect(repository.getState().operationalSubjects).toEqual([]);

    repository.deliverOutbox();
    repository.deliverOutbox({ replay: true });

    const restarted = new InMemoryLifecycleRepository(repository.snapshot());
    restarted.deliverOutbox({ replay: true });

    const state = restarted.getState();
    expect(state.operationalSubjects).toEqual([]);
    expect(state.warehouseSubjects).toEqual([
      {
        pseudoId: "subject-7",
        generationFloor: 2n,
        currentGeneration: null,
        payload: null,
      },
    ]);
    expect(state.outbox).toHaveLength(1);
    expect(state.outbox[0]).toMatchObject({
      generation: 1n,
      status: "suppressed",
      attempts: 3,
    });

    const replayAfterRestart = restarted.enqueueEvent(delayedEvent);
    expect(replayAfterRestart).toEqual(replayAfterDelete);
    expect(restarted.getState()).toEqual(state);
    expect(
      restarted.snapshot().idempotency.find(({ key }) => key === "event-1")?.result,
    ).not.toHaveProperty("payload");

    expect(() =>
      restarted.enqueueEvent({
        pseudoId: "subject-7",
        generation: 3n,
        payload: { segment: "returning" },
        idempotencyKey: "event-after-restart",
      }),
    ).toThrow(SubjectDeletedError);

    const app = buildApp({ repository: restarted });
    const rejected = await app.inject({
      method: "POST",
      url: "/v1/events",
      payload: {
        pseudoId: "subject-7",
        generation: "3",
        payload: { segment: "returning" },
        idempotencyKey: "event-via-api",
      },
    });
    expect(rejected.statusCode).toBe(410);
    expect(rejected.json()).toMatchObject({ code: "SUBJECT_DELETED" });
    await app.close();
  });

  it("repeats the same idempotent event and rejects a key collision", () => {
    const repository = new InMemoryLifecycleRepository();
    const event = {
      pseudoId: "subject-8",
      generation: 1n,
      payload: { preference: "minimal" },
      idempotencyKey: "stable-key",
    } as const;

    const first = repository.enqueueEvent(event);
    const repeated = repository.enqueueEvent(event);

    expect(first.duplicate).toBe(false);
    expect(repeated.duplicate).toBe(true);
    expect(repeated.receipt.eventId).toBe(first.receipt.eventId);
    expect(first.receipt).not.toHaveProperty("payload");
    expect(repeated.receipt).not.toHaveProperty("payload");
    expect(repository.getState().outbox).toHaveLength(1);

    expect(() =>
      repository.enqueueEvent({
        ...event,
        payload: { preference: "ornate" },
      }),
    ).toThrow(IdempotencyConflictError);
  });

  it("delivers a current generation once and exposes it through the API", async () => {
    const repository = new InMemoryLifecycleRepository();
    const app = buildApp({ repository });

    const accepted = await app.inject({
      method: "POST",
      url: "/v1/events",
      payload: {
        pseudoId: "subject-9",
        generation: "1",
        payload: { locale: "de-DE" },
        idempotencyKey: "event-current",
      },
    });
    expect(accepted.statusCode).toBe(202);

    const delivered = await app.inject({
      method: "POST",
      url: "/v1/outbox/deliver",
      payload: {},
    });
    expect(delivered.statusCode).toBe(200);
    expect(delivered.json()).toMatchObject({ applied: 1, suppressed: 0 });

    const state = await app.inject({ method: "GET", url: "/v1/state" });
    expect(state.statusCode).toBe(200);
    expect(state.json().warehouseSubjects).toEqual([
      {
        pseudoId: "subject-9",
        generationFloor: "0",
        currentGeneration: "1",
        payload: { locale: "de-DE" },
      },
    ]);

    const health = await app.inject({ method: "GET", url: "/health" });
    expect(health.json()).toEqual({ status: "ok" });
    await app.close();
  });

  it("returns HTTP 409 for an idempotency collision", async () => {
    const app = buildApp({ repository: new InMemoryLifecycleRepository() });
    const request = {
      method: "POST" as const,
      url: "/v1/events",
      payload: {
        pseudoId: "subject-10",
        generation: "1",
        payload: { value: "a" },
        idempotencyKey: "collision",
      },
    };

    expect((await app.inject(request)).statusCode).toBe(202);
    const collision = await app.inject({
      ...request,
      payload: { ...request.payload, payload: { value: "b" } },
    });
    expect(collision.statusCode).toBe(409);
    expect(collision.json()).toMatchObject({ code: "IDEMPOTENCY_CONFLICT" });
    await app.close();
  });

  it("rejects generation gaps in memory and through the API", async () => {
    const repository = new InMemoryLifecycleRepository();
    repository.enqueueEvent({
      pseudoId: "subject-11",
      generation: 1n,
      payload: { value: "first" },
      idempotencyKey: "sequence-1",
    });

    expect(() =>
      repository.enqueueEvent({
        pseudoId: "subject-11",
        generation: 3n,
        payload: { value: "gap" },
        idempotencyKey: "sequence-3",
      }),
    ).toThrow(GenerationSequenceError);

    const app = buildApp({ repository });
    const response = await app.inject({
      method: "POST",
      url: "/v1/events",
      payload: {
        pseudoId: "subject-11",
        generation: "3",
        payload: { value: "gap" },
        idempotencyKey: "sequence-3-api",
      },
    });
    expect(response.statusCode).toBe(409);
    expect(response.json()).toMatchObject({
      code: "GENERATION_SEQUENCE_ERROR",
      expectedGeneration: "2",
    });
    await app.close();
  });

  it("rejects a stale delete without writes and replays only the original delete key", () => {
    const repository = new InMemoryLifecycleRepository();
    repository.enqueueEvent({
      pseudoId: "subject-12",
      generation: 1n,
      payload: { version: 1 },
      idempotencyKey: "subject-12-event-1",
    });
    repository.enqueueEvent({
      pseudoId: "subject-12",
      generation: 2n,
      payload: { version: 2 },
      idempotencyKey: "subject-12-event-2",
    });

    const before = repository.snapshot();
    expect(() =>
      repository.deleteSubject({
        pseudoId: "subject-12",
        expectedGeneration: 1n,
        idempotencyKey: "subject-12-stale-delete",
      }),
    ).toThrow(GenerationConflictError);
    expect(repository.snapshot()).toEqual(before);

    const deleted = repository.deleteSubject({
      pseudoId: "subject-12",
      expectedGeneration: 2n,
      idempotencyKey: "subject-12-delete",
    });
    expect(deleted.receipt.generation).toBe(3n);
    expect(
      repository.deleteSubject({
        pseudoId: "subject-12",
        expectedGeneration: 2n,
        idempotencyKey: "subject-12-delete",
      }),
    ).toEqual({ ...deleted, duplicate: true });
    expect(() =>
      repository.deleteSubject({
        pseudoId: "subject-12",
        expectedGeneration: 3n,
        idempotencyKey: "subject-12-fresh-delete",
      }),
    ).toThrow(SubjectDeletedError);
    expect(repository.getState().tombstones[0]?.generation).toBe(3n);
  });

  it("round-trips generations above Number.MAX_SAFE_INTEGER as decimal strings", async () => {
    const currentGeneration = 9_007_199_254_740_993n;
    const repository = new InMemoryLifecycleRepository({
      operationalSubjects: [
        {
          pseudoId: "subject-large",
          generation: currentGeneration,
          payload: { version: "large" },
        },
      ],
      tombstones: [],
      warehouseSubjects: [],
      outbox: [],
      idempotency: [],
    });
    const app = buildApp({ repository });

    const eventResponse = await app.inject({
      method: "POST",
      url: "/v1/events",
      payload: {
        pseudoId: "subject-large",
        generation: "9007199254740994",
        payload: { version: "larger" },
        idempotencyKey: "large-event",
      },
    });
    expect(eventResponse.statusCode).toBe(202);
    expect(eventResponse.json().receipt.generation).toBe("9007199254740994");

    const deleteResponse = await app.inject({
      method: "POST",
      url: "/v1/subjects/subject-large/delete",
      payload: {
        expectedGeneration: "9007199254740994",
        idempotencyKey: "large-delete",
      },
    });
    expect(deleteResponse.statusCode).toBe(200);
    expect(deleteResponse.json().receipt.generation).toBe("9007199254740995");

    const nonCanonical = await app.inject({
      method: "POST",
      url: "/v1/events",
      payload: {
        pseudoId: "another-subject",
        generation: "01",
        payload: {},
        idempotencyKey: "non-canonical",
      },
    });
    expect(nonCanonical.statusCode).toBe(400);
    await app.close();
  });

  it("does not expose unexpected error details", async () => {
    const throwingRepository: LifecycleRepository = {
      enqueueEvent() {
        throw new Error("postgres password is secret");
      },
      deleteSubject() {
        throw new Error("not used");
      },
      deliverOutbox() {
        throw new Error("not used");
      },
      getState() {
        throw new Error("not used");
      },
      snapshot() {
        throw new Error("not used");
      },
    };
    const app = buildApp({ repository: throwingRepository });
    const response = await app.inject({
      method: "POST",
      url: "/v1/events",
      payload: {
        pseudoId: "subject-error",
        generation: "1",
        payload: {},
        idempotencyKey: "error",
      },
    });
    expect(response.statusCode).toBe(500);
    expect(response.json()).toEqual({
      code: "INTERNAL_ERROR",
      message: "Internal server error",
    });
    expect(response.body).not.toContain("postgres password");
    await app.close();
  });

  it.each([
    "alphabetic",
    "+1",
    "-1",
    "1.5",
    "",
    "01",
    "10000000000000000000",
    "9223372036854775808",
  ])("rejects malformed or out-of-range event generation %j", async (generation) => {
    const app = buildApp({ repository: new InMemoryLifecycleRepository() });
    const response = await app.inject({
      method: "POST",
      url: "/v1/events",
      payload: {
        pseudoId: "generation-validation",
        generation,
        payload: {},
        idempotencyKey: `event-${generation}`,
      },
    });
    expect(response.statusCode).toBe(400);
    expect(response.json()).toMatchObject({ code: "INVALID_REQUEST" });
    await app.close();
  });

  it.each([
    "alphabetic",
    "+1",
    "-1",
    "1.5",
    "",
    "01",
    "10000000000000000000",
    "9223372036854775808",
  ])("rejects malformed or out-of-range delete generation %j", async (expectedGeneration) => {
    const app = buildApp({ repository: new InMemoryLifecycleRepository() });
    const response = await app.inject({
      method: "POST",
      url: "/v1/subjects/generation-validation/delete",
      payload: {
        expectedGeneration,
        idempotencyKey: `delete-${expectedGeneration}`,
      },
    });
    expect(response.statusCode).toBe(400);
    expect(response.json()).toMatchObject({ code: "INVALID_REQUEST" });
    await app.close();
  });

  it("accepts the PostgreSQL bigint maximum at the HTTP boundary", async () => {
    const maximum = 9_223_372_036_854_775_807n;
    const repository = new InMemoryLifecycleRepository({
      operationalSubjects: [
        {
          pseudoId: "subject-maximum",
          generation: maximum - 1n,
          payload: {},
        },
      ],
      tombstones: [],
      warehouseSubjects: [],
      outbox: [],
      idempotency: [],
    });
    const app = buildApp({ repository });
    const event = await app.inject({
      method: "POST",
      url: "/v1/events",
      payload: {
        pseudoId: "subject-maximum",
        generation: maximum.toString(),
        payload: {},
        idempotencyKey: "maximum-event",
      },
    });
    expect(event.statusCode).toBe(202);
    expect(event.json().receipt.generation).toBe(maximum.toString());

    const deletion = await app.inject({
      method: "POST",
      url: "/v1/subjects/subject-maximum/delete",
      payload: {
        expectedGeneration: maximum.toString(),
        idempotencyKey: "maximum-delete",
      },
    });
    expect(deletion.statusCode).toBe(409);
    expect(deletion.json()).toMatchObject({ code: "GENERATION_OVERFLOW" });
    await app.close();
  });
});
