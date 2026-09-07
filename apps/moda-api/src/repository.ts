import type {
  DeleteSubjectInput,
  DeleteSubjectReceipt,
  OperationalSubject,
  OutboxRecord,
  SubjectEventInput,
  SubjectEventReceipt,
  WarehouseSubject,
} from "./domain.js";

export type EnqueueResult = {
  receipt: SubjectEventReceipt;
  duplicate: boolean;
};

export type DeleteResult = {
  receipt: DeleteSubjectReceipt;
  duplicate: boolean;
};

export type DeliveryResult = {
  examined: number;
  applied: number;
  suppressed: number;
  alreadyApplied: number;
};

export type LifecycleState = {
  operationalSubjects: OperationalSubject[];
  tombstones: DeleteSubjectReceipt[];
  warehouseSubjects: WarehouseSubject[];
  outbox: OutboxRecord[];
};

export type IdempotencySnapshot = {
  key: string;
  fingerprint: string;
  kind: "event" | "delete";
  result: SubjectEventReceipt | DeleteSubjectReceipt;
};

export type LifecycleSnapshot = LifecycleState & {
  idempotency: IdempotencySnapshot[];
};

/** PostgreSQL is the durable target; memory is for deterministic tests/demos. */
export interface LifecycleRepository {
  enqueueEvent(input: SubjectEventInput): EnqueueResult;
  deleteSubject(input: DeleteSubjectInput): DeleteResult;
  deliverOutbox(options?: { replay?: boolean }): DeliveryResult;
  getState(): LifecycleState;
  snapshot(): LifecycleSnapshot;
}
