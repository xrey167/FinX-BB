# Umsetzungsplan Lifecycle Integrity — 2026-09-07

## Ziel

Ein kleiner, vertikal pruefbarer Pfad verbindet Moda mit genau einer Lifecycle-Authority und verhindert
stale Nutzung bis zur Publication. Forschungsergebnisse bleiben als Evidenz erhalten; sie werden nicht
als Produkt- oder Neuheitsgarantie umetikettiert.

## Slices und Abhaengigkeiten

### 1. Konsolidierte, ehrliche Baseline

**Abhaengigkeit:** keine.

- Empfohlene Evidenz-Tips integrieren; fehlgeschlagene/negative Ergebnisse behalten; E52/E83/E84/E91
  ohne Ueberschreiben entkollidieren.
- README/Runner/Testzahlen und Paper-Aussagen an den realen Head anpassen.
- Ein `pull_request`-Workflow fuehrt Offline-Suite, `paper_numbers`, JSON/AST/Shell-Pruefung und spaeter
  PG-Smoke aus; Netzwerk-/GPU-Tests werden sichtbar getrennt.

**Gate:** sauberer Merge-Graph, keine verlorenen eindeutigen Commits, keine stillen Skips, reproduzierbar
gruener Offline-Lauf; Netzwerk/GPU werden explizit als verifiziert oder offen ausgewiesen.

### 2. Ein Authority-Writer und Bridge-Snapshot

**Abhaengigkeit:** Slice 1.

- Fastify validiert HTTP und spricht request-id-NDJSON mit genau einem lokalen Python-Engineprozess.
- Python allein besitzt Mutation, CAVI-Lock, Receipt-Signierung und spaeter die PG-Transaktion.
- PG Advisory Session Lock verhindert einen zweiten Writer. IDs und Generationen bleiben ueber TS als
  Dezimalstrings verlustfrei.

**Gate:** Contract-Tests fuer Korrelation, Timeout, Crash/Restart, malformed output, Duplicate Request,
zweiten Writer und BigInt-Grenzen. Kein TS-Pfad kann Lifecycle-State direkt schreiben.

### 3. Kanonischer Lifecycle plus resurrection-sicheres Delete

**Abhaengigkeit:** Slice 2.

- Zwei Zaehler, immutable Revisions/Events, namespace-weite eindeutige Heads und terminale Tombstones.
- Jede Mutation ist CAS plus Idempotency-Key; Head/Event/Outbox/Receipt sind eine PG16-Transaktion.
- Event/Ingest fuehren Generation; stale Events bleiben nach Delete, Retry, Restore und Neustart gesperrt.

**Gate:** echter PG16-Migrations-Smoke und E2E-Sequenz `enqueue g1 -> delete g2 -> deliver g1 twice ->
restart/replay`, ohne Resurrection in operativer DB, Warehouse oder Memory. Multi-Hop-Relink, Rollback,
Restore und ABA invalidieren alte Witnesses; Duplicate Keys sind unmoeglich oder kanonisch definiert.

### 4. Derived-State-, KV- und Publication-Sicherheit

**Abhaengigkeit:** Slice 3.

- Vollstaendige `PathWitness`-Lineage ueber Aliase, Pod, Policy, Principal/Tenant und positive/negative
  Dependencies.
- Live-Revalidation an realer neuronaler Consumption, vor KV-Reuse und vor Publication; request-lokaler
  Adapter-Kontext mit Exception-Cleanup.
- `BYPASS` ist exakt, `UNKNOWN` fail-closed. Streaming wird bis zum finalen Check gepuffert/abgebrochen.

**Gate:** deterministische Race-/Replay-Tests fuer stale Bank, Router, Payload, Aktivierung, KV,
Alias-Relink, Policy-Wechsel und in-flight Delete. Kein Cross-Request-Leak; unbeteiligte Pods bleiben
verwendbar; keine partiell veroeffentlichte stale Antwort.

### 5. Unabhaengiger kausaler Audit

**Abhaengigkeit:** Slice 4 fuer den realen Consumption-Pfad; Messcode bleibt autoritaetsfrei.

- Verblindeter Workspace-/J-Lens-Lauf mit aktiver positiver, NEVER/null und Bystander-Kontrolle.
- `ControlVerdict AND WorkspaceVerdict`; gefallene Kontrolle erzwingt `INCONCLUSIVE`.
- Receipts sind RFC-8785-kanonisiert, domaengetrennt gehasht, Ed25519-signiert und hashverkettet; der
  Verifier kann Authority nie erteilen.

**Gate:** Tampering/Truncation/Key-Rotation werden erkannt; ein kontrolliert positiver Live-Effekt ist
vor jedem `PASS` sichtbar. Audit-Resultat trennt Runtime-Integritaet, Store-Erreichbarkeit und kausale
Zugaenglichkeit.

### 6. Product-/Research-Freigabe

**Abhaengigkeit:** Slices 1–5.

- Commerce: explizite User-Bestaetigung, Tool-Auth, Order-Idempotenz, Saga/Compensation und Webhook-
  Reconciliation vor echtem Checkout.
- Datenschutz: Tenant/RLS, Purpose/Consent, Retention, Backup-/Parquet-/Log-/Adapter-Purge und Restore-
  Reaper mit Evidence-Receipt.
- Release-Manifest bindet Source-Commit, Dependency-Lock, Engine-/Model-Version und Resultat-Hashes.
- Claim-Matrix vergleicht externe Baselines und trennt bestanden, widerlegt und offen.

**Gate:** unabhaengiger Change-Review ohne offene P0-Findings; reproduzierbarer Artifact-Bundle; keine
Paper- oder Produktformulierung ueberschreitet die gemessene Evidenz. Patent-/Rechtsclaims bleiben bei
qualifizierter externer Pruefung.

## Explizit ausserhalb dieser Iteration

Verteilte Multi-Writer-Replicas, beliebige Freitext-/Paraphrasenpfade, Loeschung aus Basisgewichten,
`SHRED` als vollstaendige Loeschung, vollstaendige J-Space-Erfassung sowie Patent- oder Rechtsurteile.
