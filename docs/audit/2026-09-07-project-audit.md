# Projektaudit FinX-BB — 2026-09-07

## Urteil

**HOLD** fuer einen produktiven Moda-Lifecycle und eine Paper-Einreichung. Der SO-Bestand ist ein
umfangreicher, bemerkenswert selbstkritischer Forschungsprototyp; er ist noch keine einheitliche,
dauerhafte Lifecycle-Authority.

Dieses Audit bezieht sich ausschliesslich auf den Stand **`1c2c54a0af27ac26bedd2403ee06f76ae5119dc2` vor
der Konsolidierung**. Spaeter integrierte Branches und der aktuelle Head muessen gesondert verifiziert
werden. Der Status unten darf daher nicht als Testurteil ueber den konsolidierten Head gelesen werden.

## Priorisierte Findings

| Prio | ID | Status | Befund und Evidenz | Wirkung / naechstes Gate |
|---:|---|---|---|---|
| P0 | A-01 | broken | 64 Workflows, aber kein `pull_request`-Trigger; die meisten beobachten nur einzelne Forschungsbranches. Das Paper sagt dagegen, CI laufe bei jedem Push. | Kein belastbares Merge-Gate. Ein PR-Workflow muss Offline-Suite, Migration-Smoke und Integritaetschecks erzwingen. |
| P0 | A-02 | broken | README und `docs/systemarchitektur-v1.md` beschreiben Moda als Home-Base; auf der Audit-Baseline fehlen App/API, Migrationen und Moda-Tests. | Moda war auf dieser Baseline ein Entwurf, kein ausfuehrbares Produkt. |
| P0 | A-03 | broken | Die dokumentierten PG16-Tabellen `event` und `reco_log` sind nach `ts` partitioniert, ihre Primary Keys enthalten `ts` nicht; Child-/Default-Partitionen und `CREATE EXTENSION vector` fehlen. | Die Startmigration ist so nicht lauffaehig. Migration plus echter PG16-Smoke sind Pflicht. |
| P0 | A-04 | risky | `so/mvcc.py`, `so/incarnation.py` und `so/cavi.py` bilden konkurrierende Lifecycle-Wahrheiten. Experimente mutieren Store und Authority nacheinander; Restore/Recreate-Semantik widerspricht sich. | Crash oder Exception kann Split-Brain erzeugen. Genau eine transaktionale Authority muss schreiben. |
| P0 | A-05 | risky | `KnowledgeAdapterLM` haelt Request-Zustand in gemeinsamen Feldern (`_ctx`, `last_query`); Hooks lesen diese Felder, ohne durchgaengige request-lokale Lebensdauer. | Parallele Inferenz kann Nutzerzustand mischen. Request-Isolation und Exception-Cleanup muessen als Race-Test belegt werden. |
| P0 | A-06 | risky | Outbox/Warehouse-Vertrag kennt Event-ID, aber keine Lifecycle-Generation oder Tombstone-Ordnungsregel. | Ein verspaetetes Gen-1-Event kann nach Delete erneut Daten anlegen. Der erste Produktslice muss Resurrection nach Retry und Restart verhindern. |
| P0 | A-07 | risky | `MVCCStore.delete()` entfernt aktive Versionen, das Operationslog behaelt jedoch fruehere Payloads. Moda nennt Backups, Parquet, Inferenzlogs und spaetere Adapter, aber keinen vollstaendigen Purge-/Restore-Vertrag. | Unerreichbarkeit ist keine Loeschung. Retention, Restore-Reaper und abgeleitete Artefakte bleiben Go-live-Blocker. |
| P0 | A-08 | risky | `make novelty` prueft fest codierte Treffer und Dokumentkonsistenz, fuehrt aber keine Literatursuche neu aus. Einige Aussagen beruhen nur auf Metadaten. | Ein gruener Lauf beweist weder Vollstaendigkeit noch Patentneuheit. Negative Claims brauchen externe Quellenpruefung. |
| P0 | A-09 | risky | Der Agent darf Warenkorb/Order-Tools nutzen; ein belastbarer Vertrag fuer User-Bestaetigung, Tool-Auth, Idempotenz, Saga/Compensation und Webhooks fehlt. | Kein autonomer Kaufpfad vor einem expliziten Confirm- und Recovery-Gate. |
| P1 | A-10 | broken | Duplicate Keys sind erlaubt, aber `active_view`, Index, `kid_of` und exportierte Bank waehlen nicht dieselbe Praezedenz. | Derselbe gueltige Store kann mehrere Antworten liefern. Kanonische Eindeutigkeit erzwingen. |
| P1 | A-11 | broken | Abhaengigkeitsgrenzen widersprechen sich zwischen `pyproject`, Requirements und `setup.sh`; kein Lockfile, optionale Tests koennen unbemerkt fehlen. | Frische Installationen sind nicht reproduzierbar. Ein Environment-Vertrag und Skip-Inventar sind noetig. |
| P1 | A-12 | risky | Actions verwenden mutable Tags; ein Workflow besitzt `contents: write` und pusht generierte Resultate. | Supply-Chain- und Parallel-Run-Risiko. Actions pinnen und Resultat-Publishing isolieren. |
| P1 | A-13 | risky | Ergebnis-JSONs binden keinen Source-Commit; `paper_numbers` prueft Prosa gegen Records, nicht deren Herkunft. | Paper-Zahlen sind intern konsistent, aber nicht vollstaendig reproduzierbar. Artifact-Manifest mit Commit und Lock-Hash einfuehren. |
| P1 | A-14 | stale | README/`so/README.md`, Makefile, Committexte und vorhandene Experimente nennen widerspruechliche Testzahlen, Laufzeiten und Reichweiten; einzelne Links/`run.sh`-Kommandos sind veraltet. | Onboarding ist keine verlaessliche Betriebsanweisung. Zahlen aus CI generieren oder zeitlose Aussagen verwenden. |
| P1 | A-15 | risky | Stil-/Episode-Vektoren haben nicht ueberall eine Modellversion; Events und Empfehlungen keine durchgaengige Impression-/Request-ID; Feed und Live-Check widersprechen sich. | Vektorraeume und Trainingszuordnung koennen vermischt, Preise/Bestand veraltet sein. |
| P2 | A-16 | unknown | Kein vollstaendiger Consent-/Purpose-, RLS/RBAC-, KMS-, Retention- und Tenant-Auth-Vertrag. CORS plus API-Key ist kein Nutzerauth-Modell. | Datenschutz- und Mandantenreife bleiben unbewertet. |
| P2 | A-17 | unknown | Auf der Audit-Baseline wurden weder Full Suite/GPU noch Actions, PG-Migration, Moda-Build, Coverage, Lint, Types oder Security-Scan ausgefuehrt. | Keine Aussage ueber aktuelle Laufzeitreife; der konsolidierte Head braucht eine neue Evidence Map. |

## Status-Trennung

- **Broken:** A-01 bis A-03, A-10, A-11.
- **Risky:** A-04 bis A-09 sowie A-12, A-13 und A-15.
- **Stale:** A-14 und die CI-Aussage in A-01.
- **Unknown:** A-16, A-17 sowie reale Moda-Latenz, VRAM, Restore-Verhalten und Heavy-Run-Reproduzierbarkeit.
- **Healthy:** 196 Python-Dateien waren AST-lesbar, 61 JSON-Dateien parsebar und beide Shellskripte
  syntaktisch gueltig; alle exakt referenzierten Workflow-Pfade existierten. `so.paper_numbers` meldete
  43/43 interne Checks. Das Paper kennzeichnet sich als Draft und nennt Submission-Blocker offen.
  Eine Primarquellen-Stichprobe bestaetigte zentrale zitierte Arbeiten, nicht aber Vollstaendigkeit.

## Branch- und Evidenzlage

Der Audit-Commit lag 441 Commits vor `main`. Die Branch-Analyse fand zusaetzlich zu den beiden grossen
Ankern 146 nicht enthaltene Commits auf Forschungs-Tips. Neun Tips waren bereits exakt subsumiert;
zehn weitere maximale Tips tragen den empfohlenen Kern der Korrekturen und negativen Evidenz. Mehrere
Experiments-IDs kollidieren semantisch (E52, E83, E84, E91). Konsolidierung muss daher Provenienz
erhalten, kollidierende Dokumente umbenennen und darf fehlgeschlagene oder falsifizierende Resultate
nicht als Duplikate entfernen.

## Erstes Freigabe-Gate

Der kleinste produktrelevante Vertical Slice ist ein resurrection-sicherer Account-DELETE-Pfad:

1. eine einzige dauerhafte Authority mit `namespace`, stabiler ID, monotoner Generation, Status und Tombstone;
2. Event und Outbox tragen diese Generation; Delete schreibt Tombstone und Purge-Auftrag atomar;
3. Ingest verwirft jede veraltete Generation auch nach Retry und Neustart;
4. eine ausfuehrbare PG16-Migration;
5. End-to-End: Gen 1 einreihen, Gen 2 loeschen, altes Event zweimal verspaetet zustellen, neu starten und
   beweisen, dass operative, Warehouse- und Memory-Sichten leer bleiben;
6. dieser Test ist verpflichtendes `pull_request`-Gate.

Erst danach folgen Adapter-Request-Isolation, abgeleitete Lineage und Paper-/Release-Freigabe.
