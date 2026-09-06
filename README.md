# FinX-Moda

**Der AI-Fashion-Agent, der dich wirklich kennt.**

FinX-Moda (kurz: **Moda**) ist ein AI-natives Consumer-Frontend für Premium- und Luxus-Fashion-Commerce. Ein persönlicher Agent, der einen individualisierten Katalog on the fly generiert, Outfits komponiert, zu Größe und Passform berät und den Kauf wirklich abschließt — auf Live-Merchant-Infrastruktur (Genesys/Stockchain + Stripe).

Der Unterschied zu jedem GPT-Adapter: ein echter Lernkern (**finx-memory**) über L1-Graph, L2-Episoden, L3-Stilvektor — und L4-Adapter später. Lernen und vergessen sind Teil der Architektur.

## Architektur

Kanonische Referenz: [Systemarchitektur v1.0](docs/systemarchitektur-v1.md) (Stand 30.08.2026).

Invarianten: ein Schreibpfad ins Warehouse (`finx-ingest`), kein LLM im Feed-Pfad, Zonengrenzen als Code, jede Modellart versioniert.

Dieses Repository ist die Home-Base (`xrey167/FinX-BB`).

## Forschung: SO

Forschungsprojekt **SO — Modular Neural Operating System**: adressierbares, veränderbares, versionierbares und widerrufbares neuronales Wissen, das direkt an der neuronalen Berechnung teilnimmt statt nur als externer Kontext.

- [Projektstand, Vision und Architektur](docs/so-modular-neural-os.md) (Stand 02.09.2026) — Forschungsfrage, Löschung versus Unterdrückung, Provenienz, Abhängigkeitsgraph, Symlink- und Marker-Konzepte, aktuelle Architektur sowie die Ergebnisse der Experimente E-000001-A und E-000001-B.
- [Experiment- und Evidenz-Ledger](docs/so-experiment-ledger.md) (Stand 02.09.2026) — Durchbruchskriterien, Evidenzskala E0–E7, Löschmodell F0–F5, Neural-MVCC, Biomarker, Rekonstruktionsangriffe, Kausal- und Ablationstests, Stand der C-Serie; Abschnitt 31 protokolliert die in dieser Sitzung durchgeführten Experimente.
- [Sitzungsergebnisse 02.09.2026](docs/so-results-2026-09-02.md) — automatisch aus den Ergebnisdateien erzeugt: alle Messwerte, vorregistrierte Kriterien, Evidenz- und Löschstufen, Grenzen der Evidenz.
- [Was dieses Programm tatsächlich gefunden hat](docs/so-what-was-found-2026-09-04.md) (Stand 04.09.2026) —
  die Zusammenfassung: warum der Angriffs-Standard hier gebrochen ist, das Zertifikat, das über die gesamte
  Nutzlast-Domäne und über jede mögliche Anfrage quantifiziert, die drei ersten zertifizierten Löschungen,
  und was das Halten von Fakten in Zeilen messbar einbringt.
- **[Die Behauptung: Kanonisierung ist eine Löschung-Offenlegung-Dualität](docs/so-claim-erasure-disclosure-duality.md)**
  (Stand 04.09.2026) — die eine Neuheitsbehauptung dieses Programms, ihre Messung und die Vorarbeiten,
  die sie überstehen muss. Ein Pod macht einen Fakt in **einer** Löschung unerreichbar statt in k — und
  macht die Löschung selbst aus dem Store allein eindeutig identifizierbar (1.0000 gegen 0.0000), wo
  Duplikation den gesamten Schlüsselraum offen lässt. Die beiden Abschlüsse invertieren exakt.
- **[Wo ein Zugänglichkeits-Audit eines externen Speichers gelesen werden muss](docs/novelty/audit-siting-claim.md)**
  (Stand 06.09.2026) — E-000063, das zusammengesetzte Zertifikat aus Speicheroperation und kausalem
  Audit, hat zum ersten Mal gemessen: auf allen drei Seeds liefert es sein Urteil *keine
  Workspace-Spur nach der Löschung*, während seine eigene vorregistrierte Gültigkeitszeile — sieht
  das Instrument einen lebenden Pod überhaupt? — auf allen drei Seeds fällt. WSC-001 sagt warum, und
  es liegt nicht an der Basis: der Schreibvorgang an der ersten Lesestelle bewegt sich um 0.032
  dessen, was die zweite tut, und dort trennt **kein** Auslesen — auch nicht der volle
  768-dimensionale Residualstrom — lebenden Speicher von nie geschriebenem. Zwei der vier Sätze
  dieses Dokuments sind an ihren eigenen vorregistrierten Schranken zurückgezogen.
  **SIT-001 ergänzt die zweite Hälfte:** ein Audit ist nur dort zulässig, wo der Inhalt angekommen ist
  *und* die Linse noch nicht die Unembedding-Zeile ist — beim aufgezeichneten Adapter ist dieses
  **Fenster leer**, weil beide Bedingungen einander ausschließen. Verlegt man die Schreibstellen von
  Block (8, 10) auf (4, 6), **öffnet es sich** (Blöcke 6–9 auf allen drei Seeds): dieselbe
  Gültigkeitszeile, die vorher fiel, liest dort +0.77 / +0.83 / +0.87, und das Löschurteil hält
  (−0.040 / −0.009 / +0.018) — zum Preis von 0.031 bis 0.045 an Alias-Lesequalität. Eine
  Kontrollmessung verweigert die verlockendste Lesart: eine dimensionsgleiche Zufallsprojektion liest
  den Pod genauso gut wie das Audit, die Basis ist es also nicht, sondern die Stelle.
  Vorregistrierungen: [WSC-001](docs/novelty/wsc001-preregister.md),
  [SIT-001](docs/novelty/sit001-preregister.md).
- [Was hier neu ist und was nicht](docs/so-novelty-2026-09-04.md) (Stand 04.09.2026) — die Kalibrierung
  gegen den Stand der Forschung. Der Mechanismus ist Wiedererfindung (SERAC, GRACE, Larimar, SILO, LMLM,
  MUNKEY); was bleibt, ist die Prüfung: dass ein Gate auf Werten kein Löschprimitiv ist, wenn ein anderer
  Term dieselbe Nutzlast liest (E-000028), und dass ein gelerntes Gate den Rand zwischen seinen
  Trainingsklassen zertifiziert und nicht das Prädikat, das es umsetzen sollte (E-000029).
- [GPU-Protokoll E-000027](docs/so-e000027-gpu-protocol.md) — ein vorregistrierter, gestufter Plan für den
  Lauf von E5 nach E6; die erste Stufe kostet nichts, jede Stufe nennt das Ergebnis, das die Ausgaben stoppt.
- [Fahrplan](docs/so-roadmap-2026-09-02.md) — was heute belegt ist, die Lücken zur Durchbruchsdefinition, Stufen 0–6 bis zur externen Reproduktion, Abbruchkriterien.
- Experimentalcode: [`so/`](so/README.md).

### Selbst nachvollziehen, auf einem Ubuntu-Server

Es wird keine GPU gebraucht. Alle aufgezeichneten Zahlen stammen von einer Maschine mit vier CPU-Kernen.

```bash
./setup.sh          # apt-Pakete, virtuelle Umgebung, PyTorch als CPU-Build, danach die Unit-Tests
make test           # 48 Tests, etwa 10 Sekunden
make smoke          # verkleinerte Fassung der synthetischen Kette, etwa 35 Minuten
make synthetic      # die aufgezeichnete synthetische Kette, etwa 3 Stunden
make gpt2           # die Kette mit eingefrorenem GPT-2, etwa 20 Stunden, lädt GPT-2 einmalig
make demo           # eine Löschung live: Modell antwortet, eine Operation, vier Angriffe auf Zufallsniveau
make report         # baut docs/so-results-2026-09-02.md aus so/results/ neu
```

`make env` zeigt vorab Interpreter, Versionen, Threadzahl und freien Speicher. Stellschrauben sind
`PY`, `THREADS` und `SEEDS`, etwa `make gpt2 SEEDS="0"` für einen statt drei Seeds. Die gemessenen
Laufzeiten je Experiment stehen in [so/README.md](so/README.md); sie stammen aus dem Feld
`train_seconds` der Ergebnisdateien und sind damit keine Schätzung.
