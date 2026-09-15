# Konsolidierung der Forschungshistorien — 2026-09-07

## Ergebnis

Die Integrationsbranch `feat/lifecycle-integrity-consolidation` bewahrt die Git-Provenienz beider
großen Forschungsverläufe und der zehn stärksten zusätzlichen Evidenz-Tips. Es wurden keine
fehlgeschlagenen oder negativen Resultate umgeschrieben und keine Branches gelöscht.

Der Anker-Merge `ad3872b` verbindet:

- `1c2c54a` — Depression/Creative-Novelty-Verlauf;
- `46ce513` — Vision/Technical-Novelty-Verlauf.

Eine Topologieprüfung von 38 `origin/research/*`-Tips ergab neun bereits subsumierte Tips und 146
weitere eindeutige Commits außerhalb der beiden Anker. Die folgende Kernmenge integrierte 83 dieser
Commits als echte Merges:

| Evidenz-Tip | Merge-Commit | Rolle |
|---|---|---|
| `cavi-continuation-audit` | `2270b0f` | Request-Lifetime, Provenienz- und Dependency-Gegenbeispiele |
| `e84-equivariance-baseline-audit` | `fb3c990` | Korrektur: E84-Screen entspricht gewöhnlichem Late Binding |
| `gen001-token-feedback-boundary` | `578f8bf` | Generated-Token-Kontamination und deferred reads |
| `cat001-compiled-transform-state` | `f2bddc9` | Exakte FIR/CAT-Reduktionen |
| `e000088-strict-marker-reader-gate` | `30b4587` | Strikter, hash-geprüfter Marker-Reader-Gate |
| `jlens-frozen-evidence-20260905` | `41f2725` | Eingefrorene J-Lens-Evidenz und Korrektur eines schwachen Ergebnisses |
| `lcc001-lineage-certificate-boundary` | `b73375b` | Provenienz-Identifizierbarkeit und endliche Probegrenzen |
| `rbc001-consumption-boundary-completeness` | `eaa69e2` | Reentrante Consumption-/Witness-Gegenmodelle |
| `revocation-locality-mix001` | `7d41fa0` | Negative Zwei-Backbone-Evidenz zur additiven KV-Erasure |
| `rsi001-revision-sufficient-state` | `9a1813d` | Verifizierter Revision-State-Nachtrag |

Der einzige inhaltliche Kernkonflikt lag in `so/llm_adapter.py`. Die Auflösung behielt CAVIs
exception-sicheren, thread-lokalen Request-Kontext und portierte GENs `write_layer`, deferred reads,
Write-Hook und Post-Forward-Leerheitsprüfung. Zieltests für Lifetime, Token-Feedback, Write-Layer und
Injection bestanden nach der Auflösung.

## Bewusst nicht integriert

63 eindeutige Commits bleiben auf ihren Remote-Branches erhalten. Sie sind keine verlorenen
Duplikate, sondern noch nicht freigabefähig:

- semantische Experiment-ID-Kollisionen bei **E52, E83, E84 und E91**;
- unvollständig ausgeführte oder nur vorregistrierte Verläufe;
- Resultate mit gefallenen Kontrollen oder explizit falsifizierten Gates.

Vor einer zweiten Integrationswelle müssen kollidierende Dokumente umbenannt und alle Verweise
angepasst werden. Der bestehende zertifizierte E91-Verlauf darf beispielsweise nicht durch den
anderen Real-Reader-E91-Verlauf überschrieben werden.

## Integritätsregeln

1. Merge-Commits statt History-Rewrite bewahren Abstammung und negative Evidenz.
2. Gleiche Experimentnummer bedeutet nicht automatisch gleichen Versuch.
3. Ein grüner mechanischer Registry-Check ist keine neue Literaturrecherche.
4. Noch nicht ausgeführte Preregistrierungen werden nicht als Evidenz gezählt.
5. Die verbliebenen 63 Commits bleiben auffindbar, bis ihre IDs und Kontrollen sauber geklärt sind.
