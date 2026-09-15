# Lifecycle Integrity v1

## Zweck und Grenze

Dieser Vertrag vereinheitlicht Moda- und SO-Lifecycle-Semantik. Er verspricht fuer registrierte,
exakte Identitaeten atomare Zustandswechsel, stale-state-Abwehr und nachvollziehbare Receipts. Er
verspricht **nicht**, alle sprachlichen Paraphrasen zu finden, Wissen aus Basisgewichten zu entfernen,
vollstaendige neuronale Kausalitaet zu erkennen oder mit `SHRED` allein rechtliche Loeschung zu belegen.

> **Konformitaetsstatus am 2026-09-07:** Dieses Dokument ist der normative Zielvertrag, nicht die
> Beschreibung eines bereits produktiven Gesamtsystems. Die aktuelle `so/lifecycle.py`-Referenz ist
> single-process und in-memory, kennt nur registrierte exakte Aliase und signiert Audit-Records mit
> einem getrennten HMAC-SHA-256-Schluessel ohne Rotation. Die laufende Moda-API nutzt einen separaten
> TypeScript-In-Memory-Adapter; sie ist weder mit der Python-Referenz noch mit PostgreSQL verbunden.
> Die PostgreSQL-Migration und ihr Restart-/Replay-Smoke pruefen den dauerhaften Schema-Zielpfad
> separat. RFC 8785, Ed25519, Fastify-zu-Python-Bridge, produktiver PG-Adapter und externer
> Transactional-Outbox-/2PC-Nachweis sind noch nicht implementiert. Die Anforderungen unten sind als
> `MUST` fuer v1 zu lesen; ein vorhandener Test eines Teilpfads bedeutet keine Gesamtkonformitaet.

## Eine Authority, zwei Zaehler

Jeder kanonische Pod-Head lautet:

```text
(namespace, pod_id, authority_generation g, payload_revision r,
 status, payload_hash, policy_generation, commit_seq)
```

- `payload_revision r` identifiziert eine unveraenderliche Payload-Version. Neue Bytes erzeugen eine
  neue Revision; bestehende Revisionen werden nie ueberschrieben.
- `authority_generation g` steigt bei **jedem** autoritaetsrelevanten Zustandswechsel: Update, Revoke,
  Restore, Rollback, Delete, Policy-Wechsel oder Relink. Eine fruehere Generation kehrt nie zurueck.

Damit wird ABA ausgeschlossen: Rollback oder Restore darf alte Bytes wieder auswaehlen, aber nur unter
einer neuen Generation. Eine geloeschte ID ist terminal und darf nicht neu vergeben werden.

Aliases besitzen analog `(namespace, alias_id, generation, status, target_kind, target_id, commit_seq)`.
Kanonische IDs und Alias-IDs sind innerhalb ihres Namespace eindeutig.

## Zustandsmaschine

| Operation | Vorbedingung | Neuer Head | Wirkung |
|---|---|---|---|
| `CREATE` | ID nie verwendet | `g=1`, neue `r`, `ACTIVE` | Head, Revision und Event atomar |
| `UPDATE` | `ACTIVE`, erwartetes `g` stimmt | `g+1`, neue `r`, `ACTIVE` | neue immutable Payload |
| `REVOKE` | `ACTIVE`, erwartetes `g` stimmt | `g+1`, gleiche `r`, `REVOKED` | keine Aufloesung/Consumption; Retry nur ueber denselben Idempotency-Key |
| `RESTORE` | `REVOKED`, erwartetes `g` stimmt | `g+1`, gewaehltes `r`, `ACTIVE`; `null` bindet das aktuelle `r` | keine Wiederverwendung alter Witnesses |
| `ROLLBACK` | `ACTIVE`; Zielrevision gehoert zur ID | `g+1`, Ziel-`r`, `ACTIVE` | alte Bytes, neue Authority |
| `RELINK` | Alias und kompletter Zielpfad gueltig | Alias-`g+1` | alte Pfad-Witnesses stale |
| `DELETE` | nicht `DELETED`, erwartetes `g` stimmt | `g+1`, `DELETED`, Tombstone | terminal; Purge asynchron, Sperre sofort |

Jede Mutation verlangt `expected_generation` und `idempotency_key`. Dieselbe Idempotency-Key-/Request-
Kombination liefert dasselbe Resultat; abweichender Inhalt ist ein Konflikt. Ein CAS-Fehler schreibt
nichts. Head, Event, Outbox und Audit-Record committen in derselben PostgreSQL-Transaktion.

## Vollstaendiger Pfad-Witness

Ein `PathWitness` bindet nicht nur den letzten Alias, sondern jede Kante:

```text
namespace, root_alias,
[(alias_id, alias_generation, target_kind, target_id), ...],
pod_id, authority_generation, payload_revision, payload_hash,
policy_generation, principal, tenant, issued_at, expires_at, nonce, use_id
```

Die Authority liest Alias-Pfad, Pod, Policy und Abhaengigkeiten aus **einem** konsistenten Snapshot.
Jede fehlende, widerrufene, zyklische, zu tiefe oder geaenderte Kante macht den Witness ungueltig.
Capabilities binden den vollstaendigen Witness, Principal/Tenant, Nonce, Ablauf und einmalige Nutzung;
serialisierte Router, Banks, Payload-Vektoren, Aktivierungen und KV-Caches bleiben Daten, nie Authority.

## Routing: drei statt zwei Zustaende

- `BYPASS`: Anfrage liegt nach registrierter Scope-Regel ausserhalb des Memory-Pfads. Ergebnis muss dem
  No-Memory-Pfad exakt entsprechen.
- `RESOLVE(PathWitness)`: Anfrage liegt im Scope und der komplette aktuelle Pfad ist gueltig.
- `UNKNOWN`: Anfrage liegt im Scope, aber Pfad, Authority, Policy oder Freshness ist fehlend, stale oder
  fehlerhaft. `UNKNOWN` darf nie auf einen unkontrollierten Base-Model-Guess zurueckfallen.

Der MVP kennt nur registrierte exakte Aliase. Freitext-/Paraphrasen-Erkennung ist kein Bestandteil
dieses Vertrags.

## Concurrency, Consumption und Publication

1. Mutation ist CAS auf `expected_generation` in einer DB-Transaktion.
2. Resolution nimmt einen konsistenten Authority-Snapshot und erzeugt einen signierten Witness.
3. Unmittelbar an jeder tatsaechlichen neuronalen Consumption wird der **ganze** Pfad erneut geprueft.
4. Alle abgeleiteten Zustaende tragen die transitive Vereinigung ihrer positiven und negativen
   Dependencies. Auch ein Cache-Miss ist eine Dependency, weil ein spaeteres Create das Ergebnis aendert.
5. Vor KV-Wiederverwendung und vor jeder extern sichtbaren Publication erfolgt erneut ein Freshness-
   Check. Ein Effekt linearisiert vollstaendig vor oder nach einer Mutation, nie dazwischen.
6. Streaming puffert bis zur finalen Pruefung; pro Chunk braucht es eine definierte Generation. Bei
   Invalidierung wird verworfen/abgebrochen, nicht teilweise weiter publiziert.

Ein lokaler `RLock` kann den Single-Process-Prototyp schuetzen, ist aber kein verteilter Vertrag.
Reentrante Callbacks, post-read KV und Publication liegen sonst ausserhalb des Locks. Offline oder
gecachete Authority hat prinzipiell ein positives Stale-Use-Fenster; sofortiger Widerruf verlangt
aktuelle Online-Validierung oder Quieszenz.

## Receipts und Audit

Sieben Receipt-Arten: `MUTATION`, `RESOLUTION`, `CAPABILITY_ISSUANCE`, `CONSUMPTION`,
`DERIVATION`, `PUBLICATION`, `VERIFIER_RUN`. Das Envelope
enthaelt Namespace, Request-/Idempotency-ID, Commit-Sequenz, vorherigen Hash, relevanten Head/Witness,
Entscheidung, Zeit, Engine-/Code-Version und Signaturschluessel-ID. JSON wird nach RFC 8785
kanonisiert; der domaengetrennte SHA-256-Eventhash wird mit Ed25519 signiert und mit `previous_hash`
verkettet.

Ein Receipt beweist nur, dass die Engine diesen Uebergang bzw. diese Entscheidung signiert
protokolliert hat. Es beweist weder physische Erasure noch semantisches Vergessen, vollstaendige
Kausalitaet oder aktuelle Gueltigkeit ohne erneute Authority-Pruefung.

## Unabhaengiger kausaler Verifier

J-Space/J-Lens bleibt ein Audit-Instrument ausserhalb von Routing, Training und Authorization. Ein Run
vergleicht mindestens aktive positive Kontrolle, geloeschte/revokte Bedingung, NEVER/null und
unbeteiligte Bystander. `ControlVerdict` und `WorkspaceVerdict` werden erst nach verblindeter Messung
zusammengefuehrt.

- `PASS` nur, wenn **beide** Verdicts `PASS` sind und die aktive Bedingung einen vorregistriert
  materiellen kausalen Effekt zeigt.
- `FAIL`, wenn qualifizierte Evidenz eine vorregistrierte Sicherheits-/Leakage-Grenze verletzt.
- `INCONCLUSIVE`, sobald positive, Null- oder Bystander-Kontrolle, Instrumentgueltigkeit, Power oder
  Reproduzierbarkeit nicht qualifiziert. Eine gefallene positive Kontrolle darf nie zu `PASS` werden.

Runtime-Authorization haengt niemals vom Verifier ab. Der Verifier kann einen Runtime-Receipt
widerlegen oder offenlassen, aber nicht Autoritaet verleihen.

## MVP-Seam und Zielarchitektur

```text
Fastify/TypeScript -> request-id NDJSON/stdin-stdout -> genau ein Python-Engineprozess -> PostgreSQL 16
```

Fastify validiert HTTP und serialisiert IDs/Generationen als Dezimalstrings. Nur Python besitzt
Lifecycle-Mutationen, PG-Transaktion, CAVI-Lock und Receipt-Signierung. Ein PostgreSQL Advisory
**Session** Lock verhindert einen zweiten lokalen Engine-Writer. HTTP-Diagnostik schliesst kein
TOCTOU; die entscheidenden Checks bleiben an Consumption und Publication.

Der spaetere Netzwerkdienst darf den Vertrag ersetzen, nicht aufspalten: ein Schreibpfad, eine
Authority, dieselben CAS-/Witness-/Receipt-Regeln.

### PostgreSQL-16-Prinzipien

- `CREATE EXTENSION vector` ist explizit; Migrationen laufen gegen echtes PG16.
- Unpartitionierte `lifecycle_head`- und Tombstone-Tabellen sichern namespace-weite Eindeutigkeit.
- Payload-Revisions, Alias-Versionen und Lifecycle-Events sind immutable; Foreign Keys binden sie an
  stabile IDs. Head und Event werden atomar aktualisiert.
- Partitionierte Event-/Reco-Tabellen haben Unique/Primary Keys, die den Partition-Key enthalten;
  globale Idempotenz lebt in einer separaten unpartitionierten Inbox/Key-Tabelle. Es existiert eine
  Default- oder vorausgerollte Partition.
- Lineage ist eine normalisierte Dependency-Tabelle ueber Subject-Kind/ID/Generation; positive und
  negative Abhaengigkeiten sind unterscheidbar.
- Audit-Chain und Signing-Key-Metadaten sind append-only; Key-Rotation zerstoert alte Verifikation nicht.
- Outbox traegt Subject-Generation; Tombstones blockieren stale Ingest dauerhaft, auch nach Retry,
  Restore eines Backups und Consumer-Neustart.
- RLS/RBAC, Tenant/Principal, Purpose, Retention und Purge-Status sind Schema- und Policy-Felder, keine
  spaetere Applikationskonvention.

## Abnahmekern

Akzeptiert ist v1 erst, wenn Multi-Hop-Relink, ABA-Rollback/Restore, stale Alias/Pod/Policy, Replay
einer Capability, abgeleitete Aktivierung/KV, in-flight Mutation, partielle Publication, Duplicate
Keys, verspaetetes Outbox-Event nach Delete und Audit-Tampering automatisiert scheitern, waehrend
unbeteiligte Pods und exakter `BYPASS` unveraendert bleiben.
