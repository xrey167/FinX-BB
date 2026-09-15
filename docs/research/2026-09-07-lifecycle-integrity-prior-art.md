# Lifecycle Integrity: begrenzte Prior-Art-Pruefung — 2026-09-07

## Ergebnis

Die breite Architekturbehauptung ist nicht tragfaehig: externe/editierbare Speicher, gelernte Router,
versionierte Identitaeten, Widerruf, MVCC, Fencing, Capabilities, Cache-Invalidierung, kryptographische
Loeschung, hash-verkettete Logs und unabhaengige Attestation sind jeweils etablierte Bausteine.

In dieser **begrenzten, nicht vollstaendigen** Pruefung wurde keine einzelne Quelle beobachtet, die die
gesamte Konjunktion aus kanonischer Identitaet und Generation ueber externe Speicher, Alias-Pfade,
abgeleitete Aktivierungen, Replicas, KV-Caches und in-flight Publication zusammen mit einem
unabhaengigen kausalen Audit implementiert. Das ist eine offene Prueffrage, **keine Neuheitsfeststellung**.

## Naechste neuronale Vorarbeiten

| Cluster | Primaerquellen | Konsequenz |
|---|---|---|
| Parametrisches Editing | [ROME](https://arxiv.org/abs/2202.05262), [MEMIT](https://arxiv.org/abs/2210.07229), [MEND](https://arxiv.org/abs/2110.11309) | Gezieltes Aendern von Modellverhalten und seine Efficacy/Locality-Bewertung sind etabliert; Lifecycle-Integritaet darf nicht mit erfolgreichem Edit gleichgesetzt werden. |
| Externer Edit-Speicher und Routing | [SERAC](https://proceedings.mlr.press/v162/mitchell22a.html), [GRACE](https://arxiv.org/abs/2211.11031), [WISE](https://arxiv.org/abs/2405.14768), [Larimar](https://proceedings.mlr.press/v235/das24a.html), [MEMORYLLM](https://arxiv.org/abs/2402.04624) | Side memory, Keys, Router, Shards sowie selektives Aendern/Entfernen sind kein neuer Mechanismus. CAVI muss gegen diese Baselines testen, nicht gegen einen selbst gebauten Nullarm. |
| Multi-Hop, Kontext und Zeit | [MQuAKE](https://aclanthology.org/2023.emnlp-main.971/), [RippleEdits](https://aclanthology.org/2024.tacl-1.16/), [ReMaKE](https://aclanthology.org/2024.acl-long.21/), [MLaKE](https://aclanthology.org/2025.coling-main.301/), [Context-Robust KE](https://aclanthology.org/2025.findings-acl.540/), [METO](https://arxiv.org/abs/2312.05497) | Konsistenz ueber Folgerungen, Sprachen, Kontext und zeitliche Updates ist bereits eigener Evaluationsgegenstand. Exact-alias-Tests belegen keine sprachliche Vollstaendigkeit. |
| Memory-Freshness und Lineage | [STALE/CUPMem](https://arxiv.org/abs/2605.06527), [TEPA](https://arxiv.org/abs/2608.07429), [MemLineage](https://arxiv.org/abs/2605.14421), [ChronoMem](https://arxiv.org/abs/2607.27773), [stale KV](https://arxiv.org/abs/2608.15939) | Stale Memory, Revocation, zeitliche Provenienz und KV-Freshness sind benannte Problemklassen. Der moegliche Rest ist nur eine strengere End-to-End-Invariante an der realen Consumption-/Publication-Grenze. |
| Unlearning und Extraction | [TOFU](https://arxiv.org/abs/2401.06121), [MUSE](https://arxiv.org/abs/2407.06460), [Robust Knowledge Extraction](https://openreview.net/forum?id=7erlRDoaV8), [Superficial Model Editing](https://aclanthology.org/2025.acl-long.868/) | Nominale Forget-/Edit-Metriken koennen restliche Zugriffswege uebersehen. Ein gruener Store-Test ist weder Gewichtsloeschung noch robuste semantische Erasure. |

Weitere nahe Kandidaten der Sichtung sind [BABELREFT](https://aclanthology.org/2025.findings-acl.438/),
[Eywa](https://arxiv.org/abs/2605.30771) und [TOKI](https://arxiv.org/abs/2606.06240). Sie muessen in
einer claim-genauen Matrix gegen dieselben Angriffs- und Kontrollbedingungen gelesen werden; Titel-
oder Abstract-Aehnlichkeit reicht weder zum Belegen noch zum Verwerfen eines Claims.

## Naechste Systems-/Security-Vorarbeiten

| Baustein | Primaerquelle | Konsequenz |
|---|---|---|
| Namensloeschung | [POSIX `unlink`](https://pubs.opengroup.org/onlinepubs/9799919799/functions/unlink.html) | Einen Namen zu entfernen loescht nicht automatisch alle offenen Referenzen oder Bytes. Alias-Revoke und physischer Purge sind verschiedene Aussagen. |
| Generationen und Fencing | [Chubby](https://static.usenix.org/events/osdi06/tech/full_papers/burrows/burrows_html/), [Leases](https://doi.org/10.1145/74850.74870) | Generationen/Fencing und zeitlich begrenzte Autoritaet sind etabliert. Ohne Online-Pruefung oder Quieszenz bleibt ein positives Stale-Use-Fenster. |
| Delegierte Autorisierung | [Macaroons](https://www.ndss-symposium.org/ndss2014/ndss-2014-programme/macaroons-cookies-contextual-caveats-decentralized-authorization-cloud/), [OAuth Revocation](https://www.rfc-editor.org/rfc/rfc7009), [OAuth Introspection](https://www.rfc-editor.org/rfc/rfc7662) | Kontextgebundene Capabilities, Widerruf und Live-Introspection sind keine Neuheit; Offline-Tokens koennen sofortige Revocation nicht garantieren. |
| Versionierung und Cache | [PostgreSQL MVCC](https://www.postgresql.org/docs/current/mvcc-intro.html), [HTTP Caching](https://www.rfc-editor.org/rfc/rfc9111), [Cassandra Tombstones](https://cassandra.apache.org/doc/latest/cassandra/managing/operating/compaction/tombstones.html) | Snapshot-Sicht, Invalidierung und Tombstones sind Standard. Die Produktfrage ist korrekte Kopplung ueber alle abgeleiteten Verbraucher. |
| Integritaetslogs und Attestation | [Venti](https://www.usenix.org/events/fast02/quinlan/quinlan_html/), [Certificate Transparency](https://www.rfc-editor.org/rfc/rfc9162), [RATS](https://www.rfc-editor.org/rfc/rfc9334), [TUF](https://theupdateframework.github.io/specification/latest/) | Hashketten, signierte Metadaten und Attestation belegen Integritaet/Historie eines Records, nicht semantisches Vergessen oder aktuelle Authority. |
| Kryptographische Loeschung | [NIST SP 800-88r2](https://doi.org/10.6028/NIST.SP.800-88r2) | Key Destruction ist etablierte Sanitization-Technik; sie deckt keine Klartextkopien, Ableitungen oder Modellgewichte automatisch ab. |
| Retrieval und Vektordatenbanken | [RAG](https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf), [V3DB](https://arxiv.org/html/2603.03065v2) | Retrieval-augmentierte Inferenz und versionierte Vektordatenhaltung sind direkte Vergleichspunkte; ein eigener Store ist keine Neuheit. |
| Revocable Storage | [Proof-carrying, revocable file system](https://doi.org/10.1007/978-3-642-29963-6_5) | Beweis-/Policy-tragender Widerruf im Storage ist Vorarbeit; ein Neural-Receipt muss eine spezifische zusaetzliche Consumption-Eigenschaft zeigen. |

## Widersprueche und claim-killing Tests

1. Ein Pod-only Versionscheck besteht Alias-Relink-/ABA-Angriffe genauso gut wie der volle Pfad: dann
   traegt Alias-Lineage keine eigenstaendige Aussage.
2. Gewoehnliche Versionstags plus Cache-Invalidierung verhindern Aktivierungs-/KV-Replay vollstaendig:
   dann ist CAVI-N Systems-Hygiene, keine besondere neuronale Eigenschaft.
3. Stale post-authorization Neural State ist nicht kausal replaybar: dann fehlt der behauptete
   Angriff, auch wenn der Guard formal sauber ist.
4. Exakter `BYPASS` benoetigt versteckte Memory-Injektion oder veraendert das Basismodell: dann faellt
   die Selective-Locality-Garantie.
5. Positive Kontrolle oder NEVER/Bystander-Kontrolle des kausalen Audits faellt: Ergebnis ist
   `INCONCLUSIVE`, nicht Loeschbeleg.
6. Ein Offline-Witness soll sofortigen Widerruf garantieren: das widerspricht dem unvermeidlichen
   Stale-Fenster ohne aktuelle Authority oder Quieszenz.
7. Ein signiertes/hash-verkettetes Receipt soll physische oder semantische Loeschung beweisen: Logs
   belegen den protokollierten Vorgang, nicht die Abwesenheit aller Kopien und Effekte.

## Vorlaeufige Position

Die belastbare Engineering-These ist: **Autoritaets-Lineage muss jede wiederverwendbare, aus Memory
abgeleitete Repraesentation bis an Consumption und Publication begleiten und dort live validiert
werden.** Der Forschungswert liegt vorerst im adversarialen Audit dieser These und in negativen
Resultaten. Die Architektur selbst ist eine Zusammensetzung bekannter Muster.

Dies ist keine systematische oder erschoepfende Suche, keine Patent-Freedom-to-Operate-Analyse und
keine Rechtsberatung. Vor Patent-, Publikations- oder Marktclaim sind claim charts, aktuelle
Patentdatenbanken, Volltextpruefung und qualifizierte juristische Beratung erforderlich.
