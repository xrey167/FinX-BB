import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

const migration = readFileSync(
  new URL("../../../db/migrations/0001_moda.sql", import.meta.url),
  "utf8",
);

describe("PostgreSQL privilege boundary", () => {
  it("uses named constraints where RETURNS TABLE output names could be ambiguous", () => {
    expect(migration).not.toMatch(/ON CONFLICT\s*\(\s*pseudo_id\s*\)/i);
    expect(
      migration.match(
        /ON CONFLICT ON CONSTRAINT moda_subject_lifecycle_pkey/gi,
      ),
    ).toHaveLength(2);
    expect(
      migration.match(
        /ON CONFLICT ON CONSTRAINT moda_operational_subjects_pkey/gi,
      ),
    ).toHaveLength(1);
    expect(
      migration.match(
        /ON CONFLICT ON CONSTRAINT moda_warehouse_subjects_pkey/gi,
      ),
    ).toHaveLength(2);
  });

  it("creates a non-login role with no direct table privileges", () => {
    expect(migration).toMatch(/CREATE ROLE moda_app\s+NOLOGIN/i);
    expect(migration).toMatch(
      /REVOKE ALL PRIVILEGES\s+ON TABLE[\s\S]+?FROM moda_app;/i,
    );
    expect(migration).not.toMatch(/GRANT[\s\S]+?ON TABLE[\s\S]+?TO moda_app;/i);
    expect(migration).toMatch(/REVOKE CREATE ON SCHEMA public FROM PUBLIC;/i);
    expect(
      migration.match(
        /GRANT EXECUTE ON FUNCTION public\.[^(]+\([^)]+\) TO moda_app;/gi,
      ),
    ).toHaveLength(3);
  });

  it.each([
    "moda_enqueue_subject_event",
    "moda_tombstone_subject",
    "moda_apply_outbox",
  ])("exposes only a pinned SECURITY DEFINER function: %s", (functionName) => {
    const definition = new RegExp(
      `CREATE OR REPLACE FUNCTION ${functionName}\\([\\s\\S]+?SECURITY DEFINER\\s+SET search_path = pg_catalog, public, pg_temp`,
      "i",
    );
    expect(migration).toMatch(definition);
    expect(migration).toMatch(
      new RegExp(
        `REVOKE EXECUTE ON FUNCTION public\\.${functionName}\\([^)]+\\) FROM PUBLIC;`,
        "i",
      ),
    );
    expect(migration).toMatch(
      new RegExp(
        `GRANT EXECUTE ON FUNCTION public\\.${functionName}\\([^)]+\\) TO moda_app;`,
        "i",
      ),
    );
  });
});
