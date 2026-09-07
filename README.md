# FinX-BB

FinX-BB verbindet zwei Arbeitsstränge:

- **FinX-Moda**: ein ausführbarer, lokaler Fashion-Commerce-Prototyp.
- **SO**: ein Forschungsprogramm zu versionierbarem und widerrufbarem neuronalen Wissen.

> **Status:** Forschungs- und Engineering-Prototyp, kein produktionsreifer Commerce- oder
> Löschdienst. Checkout ist simuliert; die Lifecycle-API nutzt standardmäßig In-Memory-State. Die
> dokumentierte PostgreSQL-Migration ist noch nicht an den HTTP-Prozess angebunden. Receipts und
> Tests belegen weder physische Löschung noch semantisches Vergessen oder Patentneuheit.

## Moda lokal starten

Alles in einem Befehl (empfohlen: Node.js 24.15+ innerhalb der 24er-Linie; ebenfalls
unterstuetzt sind Node.js 22.22.2+ innerhalb der 22er-Linie sowie Node.js 26+):

```bash
./run-moda.sh
```

Der Launcher installiert die beiden Lockfiles, baut API und Storefront, startet beide nur auf
Loopback, prüft ihre Erreichbarkeit und öffnet die UI auf einem lokalen Desktop. `Ctrl+C` beendet
beide Prozesse. Mit `./run-moda.sh --verify` laufen vorher zusätzlich alle App-Tests; Ports lassen
sich über `API_PORT` und `WEB_PORT` ändern.

Einzelstart für die Entwicklung:

Frontend (React, Vite, TypeScript):

```bash
cd apps/moda-web
npm ci
npm test
npm run dev -- --host 127.0.0.1
```

Lifecycle-API (Fastify, Zod, TypeScript):

```bash
cd apps/moda-api
npm ci
npm test
npm run dev
```

Der Web-Prototyp enthält Kuratierungen, Varianten, Favoriten, Warenkorb und einen simulierten
Checkout. Die API demonstriert separat einen resurrection-sicheren Event-/Delete-Pfad mit
terminalem Tombstone, lückenlosen Generationen, Idempotenz und stale Outbox-Unterdrückung. Eine
Produktintegration zwischen beiden Apps ist bewusst noch nicht behauptet.

- [Moda-Web-Dokumentation](apps/moda-web/README.md)
- [Lifecycle-API-Dokumentation](apps/moda-api/README.md)
- [Ausführbare PostgreSQL-16-Migration](db/migrations/0001_moda.sql)
- [Desktop-Konzept](docs/design/moda-desktop-concept.png) und
  [Mobile-Konzept](docs/design/moda-mobile-concept.png)
- [Herkunft und Freigabegrenzen der Design-Assets](docs/design/README.md)

## Lifecycle-Integrität

Die aktuelle Engineering-Grenze ist ein einziger Authority-Vertrag: Payload-Revision und monotone
Authority-Generation sind getrennt; Alias-Pfade tragen vollständige Witnesses; in-scope Fehler sind
`UNKNOWN` statt stiller Modell-Fallback; Consumption und Publication müssen Freshness live prüfen.

- [Lifecycle Integrity v1](docs/architecture/lifecycle-integrity-v1.md)
- [Umsetzungsplan](docs/keystone/tasks/2026-09-07-lifecycle-integrity.md)
- [Projektaudit der Vorkonsolidierungs-Baseline](docs/audit/2026-09-07-project-audit.md)
- [Konsolidierungsprotokoll](docs/research/2026-09-07-branch-consolidation.md)

Die ältere [Systemarchitektur v1.0](docs/systemarchitektur-v1.md) bleibt als Produkt-Zielbild
erhalten. Bei Schema- oder Lifecycle-Widersprüchen gelten die ausführbare Migration und der neue
Lifecycle-Vertrag als aktuelle Referenz.

## SO-Forschung

SO untersucht adressierbares, veränderbares und widerrufbares Wissen, das an neuronaler Berechnung
teilnimmt. Die konsolidierte Evidenz enthält ausdrücklich auch falsifizierende und negative
Resultate. Die aktuelle Prior-Art-Prüfung trägt **keine breite Neuheitsbehauptung**: Editing,
externe Speicher, MVCC, Fencing, Capabilities, Tombstones, Cache-Invalidierung und Integritätslogs
sind etablierte Bausteine.

- [Begrenzte Lifecycle-Prior-Art-Prüfung](docs/research/2026-09-07-lifecycle-integrity-prior-art.md)
- [Was das Programm tatsächlich gefunden hat](docs/so-what-was-found-2026-09-04.md)
- [Experiment- und Evidenz-Ledger](docs/so-experiment-ledger.md)
- [Paper-Draft mit offenen Submission-Blockern](docs/paper/deletion-certificates-draft-2026-09-06.md)
- [Experimentalcode und Reproduktionsziele](so/README.md)

Historische Claim-Dokumente bleiben als Provenienz erhalten; sie sind im Licht späterer Kontrollen,
Korrekturen und des aktuellen Audits zu lesen.

## SO reproduzieren (Python 3.11+)

```bash
./setup.sh                 # virtuelle Umgebung, deklarierte Dependencies, Offline-Suite
make test                  # Offline-Unit- und Contract-Tests; keine Modelldownloads
make test-network          # getrennte modellgestützte Tests; lädt/liest GPT-2
make papernums             # Paper-Zahlen gegen aufgezeichnete Resultate prüfen
make auditinstr            # nummerierte E-Serien-Experimente auf zwei Defektklassen prüfen
make novelty               # Claim-Registry prüfen; keine neue Web-Literatursuche
```

Die schweren Targets `make smoke`, `make synthetic` und `make gpt2` trainieren beziehungsweise
laden Modelle und sind in [so/README.md](so/README.md) beschrieben. `make env` zeigt Interpreter,
Versionen, Threadzahl und freien Speicher. `PY`, `THREADS` und `SEEDS` sind überschreibbar.
