export type JsonValue =
  | null
  | boolean
  | number
  | string
  | JsonValue[]
  | { [key: string]: JsonValue };

export const MAX_GENERATION = 9_223_372_036_854_775_807n;

export type SubjectEventInput = {
  pseudoId: string;
  generation: bigint;
  payload: JsonValue;
  idempotencyKey: string;
};

export type DeleteSubjectInput = {
  pseudoId: string;
  expectedGeneration: bigint;
  idempotencyKey: string;
};

export type SubjectEventReceipt = {
  eventId: string;
  pseudoId: string;
  generation: bigint;
  ts: string;
};

export type DeleteSubjectReceipt = {
  pseudoId: string;
  generation: bigint;
  ts: string;
};

export type OperationalSubject = {
  pseudoId: string;
  generation: bigint;
  payload: JsonValue;
};

export type WarehouseSubject = {
  pseudoId: string;
  generationFloor: bigint;
  currentGeneration: bigint | null;
  payload: JsonValue | null;
};

export type OutboxRecord = {
  outboxId: string;
  eventId: string;
  pseudoId: string;
  generation: bigint;
  payload: JsonValue;
  idempotencyKey: string;
  ts: string;
  status: "queued" | "applied" | "suppressed";
  attempts: number;
};

export class IdempotencyConflictError extends Error {
  readonly code = "IDEMPOTENCY_CONFLICT";

  constructor(readonly idempotencyKey: string) {
    super(`Idempotency key '${idempotencyKey}' was already used for another command`);
    this.name = "IdempotencyConflictError";
  }
}

export class GenerationSequenceError extends Error {
  readonly code = "GENERATION_SEQUENCE_ERROR";

  constructor(
    readonly pseudoId: string,
    readonly generation: bigint,
    readonly expectedGeneration: bigint,
  ) {
    super(
      `Generation ${generation} for '${pseudoId}' must equal ${expectedGeneration}`,
    );
    this.name = "GenerationSequenceError";
  }
}

export class GenerationConflictError extends Error {
  readonly code = "GENERATION_CONFLICT";

  constructor(
    readonly pseudoId: string,
    readonly expectedGeneration: bigint,
    readonly actualGeneration: bigint,
  ) {
    super(
      `Expected generation ${expectedGeneration} for '${pseudoId}', actual generation is ${actualGeneration}`,
    );
    this.name = "GenerationConflictError";
  }
}

export class GenerationOverflowError extends Error {
  readonly code = "GENERATION_OVERFLOW";

  constructor(readonly pseudoId: string) {
    super(`Generation for '${pseudoId}' exceeds the PostgreSQL bigint range`);
    this.name = "GenerationOverflowError";
  }
}

export class SubjectDeletedError extends Error {
  readonly code = "SUBJECT_DELETED";

  constructor(readonly pseudoId: string) {
    super(`Subject '${pseudoId}' is terminally deleted`);
    this.name = "SubjectDeletedError";
  }
}
