# FinX-Moda

**Der AI-Fashion-Agent, der dich wirklich kennt.**

FinX-Moda (kurz: **Moda**) ist ein AI-natives Consumer-Frontend für Premium- und Luxus-Fashion-Commerce. Ein persönlicher Agent, der einen individualisierten Katalog on the fly generiert, Outfits komponiert, zu Größe und Passform berät und den Kauf wirklich abschließt — auf Live-Merchant-Infrastruktur (Genesys/Stockchain + Stripe).

Der Unterschied zu jedem GPT-Adapter: ein echter Lernkern (**finx-memory**) über L1-Graph, L2-Episoden, L3-Stilvektor — und L4-Adapter später. Lernen und vergessen sind Teil der Architektur.

## Architektur

Kanonische Referenz: [Systemarchitektur v1.0](docs/systemarchitektur-v1.md) (Stand 30.08.2026).

Invarianten: ein Schreibpfad ins Warehouse (`finx-ingest`), kein LLM im Feed-Pfad, Zonengrenzen als Code, jede Modellart versioniert.

## Forschung: SO

Begleitende Forschungsrichtung zum Lernkern: [SO — Modular Neural Operating System](docs/so-modular-neural-os.md) (Stand 02.09.2026). Projektstand, Vision und Architektur für adressierbares, veränderbares, versionierbares und widerrufbares neuronales Wissen, das direkt an der neuronalen Berechnung teilnimmt — inklusive der Ergebnisse der Experimente E-000001-A und E-000001-B.

Dieses Repository ist die Home-Base (`xrey167/FinX-BB`).
