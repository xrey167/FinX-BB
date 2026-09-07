import Fastify, { type FastifyInstance } from "fastify";
import { z } from "zod";

import {
  GenerationConflictError,
  GenerationOverflowError,
  GenerationSequenceError,
  IdempotencyConflictError,
  MAX_GENERATION,
  SubjectDeletedError,
  type JsonValue,
} from "./domain.js";
import { InMemoryLifecycleRepository } from "./in-memory-repository.js";
import type {
  DeleteResult,
  EnqueueResult,
  LifecycleRepository,
  LifecycleState,
} from "./repository.js";

const canonicalDecimal = /^(0|[1-9][0-9]*)$/;
const maximumGenerationDecimal = MAX_GENERATION.toString();

function isWithinPostgresBigint(value: string): boolean {
  if (!canonicalDecimal.test(value)) return false;
  return (
    value.length < maximumGenerationDecimal.length ||
    (value.length === maximumGenerationDecimal.length &&
      value <= maximumGenerationDecimal)
  );
}

const generationSchema = z
  .string()
  .regex(canonicalDecimal, "must be a canonical unsigned decimal string")
  .refine(isWithinPostgresBigint, "exceeds bigint range");
const positiveGenerationSchema = generationSchema.refine(
  (value) => value !== "0",
  "must be greater than zero",
);

const eventSchema = z
  .object({
    pseudoId: z.string().min(1).max(200),
    generation: positiveGenerationSchema,
    payload: z.json(),
    idempotencyKey: z.string().min(1).max(200),
  })
  .strict();
const deleteSchema = z
  .object({
    expectedGeneration: generationSchema,
    idempotencyKey: z.string().min(1).max(200),
  })
  .strict();
const deliverySchema = z.object({ replay: z.boolean().optional() }).strict();

type AppOptions = { repository?: LifecycleRepository };

function parse<T>(schema: z.ZodType<T>, value: unknown): T {
  const result = schema.safeParse(value);
  if (!result.success) {
    const error = new Error("Invalid request body") as Error & {
      statusCode: number;
      details: z.core.$ZodIssue[];
    };
    error.statusCode = 400;
    error.details = result.error.issues;
    throw error;
  }
  return result.data;
}

function serializeEnqueue(result: EnqueueResult) {
  return {
    receipt: {
      ...result.receipt,
      generation: result.receipt.generation.toString(),
    },
    duplicate: result.duplicate,
  };
}

function serializeDelete(result: DeleteResult) {
  return {
    receipt: {
      ...result.receipt,
      generation: result.receipt.generation.toString(),
    },
    duplicate: result.duplicate,
  };
}

function serializeState(state: LifecycleState) {
  return {
    operationalSubjects: state.operationalSubjects.map((subject) => ({
      ...subject,
      generation: subject.generation.toString(),
    })),
    tombstones: state.tombstones.map((tombstone) => ({
      ...tombstone,
      generation: tombstone.generation.toString(),
    })),
    warehouseSubjects: state.warehouseSubjects.map((subject) => ({
      ...subject,
      generationFloor: subject.generationFloor.toString(),
      currentGeneration: subject.currentGeneration?.toString() ?? null,
    })),
    outbox: state.outbox.map((record) => ({
      ...record,
      generation: record.generation.toString(),
    })),
  };
}

export function buildApp(options: AppOptions = {}): FastifyInstance {
  const repository = options.repository ?? new InMemoryLifecycleRepository();
  const app = Fastify({ logger: false });

  app.setErrorHandler((error, _request, reply) => {
    if (error instanceof IdempotencyConflictError) {
      return reply.status(409).send({ code: error.code, message: error.message });
    }
    if (error instanceof GenerationSequenceError) {
      return reply.status(409).send({
        code: error.code,
        message: error.message,
        expectedGeneration: error.expectedGeneration.toString(),
      });
    }
    if (error instanceof GenerationConflictError) {
      return reply.status(409).send({
        code: error.code,
        message: error.message,
        expectedGeneration: error.expectedGeneration.toString(),
        actualGeneration: error.actualGeneration.toString(),
      });
    }
    if (error instanceof GenerationOverflowError) {
      return reply.status(409).send({ code: error.code, message: error.message });
    }
    if (error instanceof SubjectDeletedError) {
      return reply.status(410).send({ code: error.code, message: error.message });
    }

    const candidate = error as Error & { statusCode?: number; details?: unknown };
    if (candidate.statusCode === 400) {
      return reply.status(400).send({
        code: "INVALID_REQUEST",
        message: candidate.message,
        ...(candidate.details ? { details: candidate.details } : {}),
      });
    }
    return reply.status(500).send({
      code: "INTERNAL_ERROR",
      message: "Internal server error",
    });
  });

  app.get("/health", async () => ({ status: "ok" }));

  app.post("/v1/events", async (request, reply) => {
    const body = parse(eventSchema, request.body);
    const result = repository.enqueueEvent({
      pseudoId: body.pseudoId,
      generation: BigInt(body.generation),
      payload: body.payload as JsonValue,
      idempotencyKey: body.idempotencyKey,
    });
    return reply
      .status(result.duplicate ? 200 : 202)
      .send(serializeEnqueue(result));
  });

  app.post("/v1/subjects/:pseudoId/delete", async (request, reply) => {
    const params = parse(z.object({ pseudoId: z.string().min(1).max(200) }), request.params);
    const body = parse(deleteSchema, request.body);
    return reply.status(200).send(
      serializeDelete(
        repository.deleteSubject({
          pseudoId: params.pseudoId,
          expectedGeneration: BigInt(body.expectedGeneration),
          idempotencyKey: body.idempotencyKey,
        }),
      ),
    );
  });

  app.post("/v1/outbox/deliver", async (request) => {
    const body = parse(deliverySchema, request.body ?? {});
    return repository.deliverOutbox(
      body.replay === undefined ? {} : { replay: body.replay },
    );
  });

  app.get("/v1/state", async () => serializeState(repository.getState()));

  return app;
}
