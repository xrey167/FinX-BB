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
make report         # baut docs/so-results-2026-09-02.md aus so/results/ neu
```

`make env` zeigt vorab Interpreter, Versionen, Threadzahl und freien Speicher. Stellschrauben sind
`PY`, `THREADS` und `SEEDS`, etwa `make gpt2 SEEDS="0"` für einen statt drei Seeds. Die gemessenen
Laufzeiten je Experiment stehen in [so/README.md](so/README.md); sie stammen aus dem Feld
`train_seconds` der Ergebnisdateien und sind damit keine Schätzung.
