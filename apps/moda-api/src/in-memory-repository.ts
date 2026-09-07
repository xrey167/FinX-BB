import { createHash, randomUUID } from "node:crypto";

import {
  GenerationConflictError,
  GenerationOverflowError,
  GenerationSequenceError,
  IdempotencyConflictError,
  MAX_GENERATION,
  SubjectDeletedError,
  type DeleteSubjectInput,
  type DeleteSubjectReceipt,
  type JsonValue,
  type OperationalSubject,
  type OutboxRecord,
  type SubjectEventInput,
  type SubjectEventReceipt,
  type WarehouseSubject,
} from "./domain.js";
import type {
  DeleteResult,
  DeliveryResult,
  EnqueueResult,
  IdempotencySnapshot,
  LifecycleRepository,
  LifecycleSnapshot,
  LifecycleState,
} from "./repository.js";

export {
  GenerationConflictError,
  GenerationOverflowError,
  GenerationSequenceError,
  IdempotencyConflictError,
  SubjectDeletedError,
} from "./domain.js";

type LifecycleHead = {
  generation: bigint;
  tombstoned: boolean;
  ts: string;
};

const clone = <T>(value: T): T => structuredClone(value);

function canonicalJson(value: JsonValue): string {
  if (value === null || typeof value !== "object") {
    const encoded = JSON.stringify(value);
    if (encoded === undefined) throw new TypeError("Value is not valid JSON");
    return encoded;
  }
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  const entries = Object.entries(value).sort(([left], [right]) =>
    left.localeCompare(right),
  );
  return `{${entries
    .map(([key, child]) => `${JSON.stringify(key)}:${canonicalJson(child)}`)
    .join(",")}}`;
}

function fingerprint(command: JsonValue): string {
  return createHash("sha256").update(canonicalJson(command)).digest("hex");
}

function byPseudoId<T extends { pseudoId: string }>(left: T, right: T): number {
  return left.pseudoId.localeCompare(right.pseudoId);
}

export class InMemoryLifecycleRepository implements LifecycleRepository {
  private readonly lifecycle = new Map<string, LifecycleHead>();
  private readonly operational = new Map<string, OperationalSubject>();
  private readonly warehouse = new Map<string, WarehouseSubject>();
  private readonly idempotency = new Map<string, IdempotencySnapshot>();
  private readonly outbox: OutboxRecord[] = [];

  constructor(snapshot?: LifecycleSnapshot) {
    if (!snapshot) return;
    for (const subject of snapshot.operationalSubjects) {
      this.operational.set(subject.pseudoId, clone(subject));
      this.lifecycle.set(subject.pseudoId, {
        generation: subject.generation,
        tombstoned: false,
        ts: new Date(0).toISOString(),
      });
    }
    for (const tombstone of snapshot.tombstones) {
      this.lifecycle.set(tombstone.pseudoId, {
        generation: tombstone.generation,
        tombstoned: true,
        ts: tombstone.ts,
      });
    }
    for (const subject of snapshot.warehouseSubjects) {
      this.warehouse.set(subject.pseudoId, clone(subject));
    }
    this.outbox.push(...clone(snapshot.outbox));
    for (const record of snapshot.idempotency) {
      this.idempotency.set(record.key, clone(record));
    }
  }

  enqueueEvent(input: SubjectEventInput): EnqueueResult {
    const commandFingerprint = fingerprint({
      kind: "event",
      pseudoId: input.pseudoId,
      generation: input.generation.toString(),
      payload: input.payload,
    });
    const duplicate = this.findIdempotentEvent(
      input.idempotencyKey,
      commandFingerprint,
    );
    if (duplicate) return { receipt: duplicate, duplicate: true };

    const lifecycleHead = this.lifecycle.get(input.pseudoId);
    if (lifecycleHead?.tombstoned) throw new SubjectDeletedError(input.pseudoId);

    const expectedGeneration = (lifecycleHead?.generation ?? 0n) + 1n;
    if (input.generation !== expectedGeneration) {
      throw new GenerationSequenceError(
        input.pseudoId,
        input.generation,
        expectedGeneration,
      );
    }
    if (input.generation > MAX_GENERATION) {
      throw new GenerationOverflowError(input.pseudoId);
    }

    const ts = new Date().toISOString();
    const receipt: SubjectEventReceipt = {
      eventId: randomUUID(),
      pseudoId: input.pseudoId,
      generation: input.generation,
      ts,
    };
    const outbox: OutboxRecord = {
      outboxId: randomUUID(),
      eventId: receipt.eventId,
      pseudoId: input.pseudoId,
      generation: input.generation,
      payload: clone(input.payload),
      idempotencyKey: input.idempotencyKey,
      ts,
      status: "queued",
      attempts: 0,
    };

    this.lifecycle.set(input.pseudoId, {
      generation: input.generation,
      tombstoned: false,
      ts,
    });
    this.operational.set(input.pseudoId, {
      pseudoId: input.pseudoId,
      generation: input.generation,
      payload: clone(input.payload),
    });
    this.outbox.push(outbox);
    this.idempotency.set(input.idempotencyKey, {
      key: input.idempotencyKey,
      fingerprint: commandFingerprint,
      kind: "event",
      result: clone(receipt),
    });
    return { receipt: clone(receipt), duplicate: false };
  }

  deleteSubject(input: DeleteSubjectInput): DeleteResult {
    const commandFingerprint = fingerprint({
      kind: "delete",
      pseudoId: input.pseudoId,
      expectedGeneration: input.expectedGeneration.toString(),
    });
    const duplicate = this.findIdempotentDelete(
      input.idempotencyKey,
      commandFingerprint,
    );
    if (duplicate) return { receipt: duplicate, duplicate: true };

    const lifecycleHead = this.lifecycle.get(input.pseudoId);
    if (lifecycleHead?.tombstoned) throw new SubjectDeletedError(input.pseudoId);

    const actualGeneration = lifecycleHead?.generation ?? 0n;
    if (input.expectedGeneration !== actualGeneration) {
      throw new GenerationConflictError(
        input.pseudoId,
        input.expectedGeneration,
        actualGeneration,
      );
    }
    if (actualGeneration === MAX_GENERATION) {
      throw new GenerationOverflowError(input.pseudoId);
    }

    const generation = actualGeneration + 1n;
    const ts = new Date().toISOString();
    const receipt: DeleteSubjectReceipt = {
      pseudoId: input.pseudoId,
      generation,
      ts,
    };
    this.lifecycle.set(input.pseudoId, { generation, tombstoned: true, ts });
    this.operational.delete(input.pseudoId);
    this.warehouse.set(input.pseudoId, {
      pseudoId: input.pseudoId,
      generationFloor: generation,
      currentGeneration: null,
      payload: null,
    });
    this.idempotency.set(input.idempotencyKey, {
      key: input.idempotencyKey,
      fingerprint: commandFingerprint,
      kind: "delete",
      result: clone(receipt),
    });
    return { receipt: clone(receipt), duplicate: false };
  }

  deliverOutbox(options: { replay?: boolean } = {}): DeliveryResult {
    const result: DeliveryResult = {
      examined: 0,
      applied: 0,
      suppressed: 0,
      alreadyApplied: 0,
    };
    for (const record of this.outbox) {
      if (!options.replay && record.status !== "queued") continue;
      result.examined += 1;
      record.attempts += 1;
      const current = this.warehouse.get(record.pseudoId) ?? {
        pseudoId: record.pseudoId,
        generationFloor: 0n,
        currentGeneration: null,
        payload: null,
      };
      if (record.generation <= current.generationFloor) {
        record.status = "suppressed";
        result.suppressed += 1;
      } else if (record.generation === current.currentGeneration) {
        record.status = "applied";
        result.alreadyApplied += 1;
      } else if (
        current.currentGeneration !== null &&
        record.generation < current.currentGeneration
      ) {
        record.status = "suppressed";
        result.suppressed += 1;
      } else {
        this.warehouse.set(record.pseudoId, {
          pseudoId: record.pseudoId,
          generationFloor: current.generationFloor,
          currentGeneration: record.generation,
          payload: clone(record.payload),
        });
        record.status = "applied";
        result.applied += 1;
      }
    }
    return result;
  }

  getState(): LifecycleState {
    const tombstones = [...this.lifecycle.entries()]
      .filter(([, head]) => head.tombstoned)
      .map(([pseudoId, head]) => ({
        pseudoId,
        generation: head.generation,
        ts: head.ts,
      }));
    return clone({
      operationalSubjects: [...this.operational.values()].sort(byPseudoId),
      tombstones: tombstones.sort(byPseudoId),
      warehouseSubjects: [...this.warehouse.values()].sort(byPseudoId),
      outbox: [...this.outbox],
    });
  }

  snapshot(): LifecycleSnapshot {
    return clone({
      ...this.getState(),
      idempotency: [...this.idempotency.values()].sort((left, right) =>
        left.key.localeCompare(right.key),
      ),
    });
  }

  private findIdempotentEvent(
    key: string,
    commandFingerprint: string,
  ): SubjectEventReceipt | undefined {
    const existing = this.idempotency.get(key);
    if (!existing) return undefined;
    if (existing.fingerprint !== commandFingerprint || existing.kind !== "event") {
      throw new IdempotencyConflictError(key);
    }
    return clone(existing.result as SubjectEventReceipt);
  }

  private findIdempotentDelete(
    key: string,
    commandFingerprint: string,
  ): DeleteSubjectReceipt | undefined {
    const existing = this.idempotency.get(key);
    if (!existing) return undefined;
    if (existing.fingerprint !== commandFingerprint || existing.kind !== "delete") {
      throw new IdempotencyConflictError(key);
    }
    return clone(existing.result as DeleteSubjectReceipt);
  }
}
