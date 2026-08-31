# FinX-Moda Systemarchitektur v1.0

**Projekt:** FinX-BB · FinX-Moda  
**Stand:** 30.08.2026 · Agent + finx-memory  
**Stack:** TypeScript + Python  
**Inferenz:** self-hosted, Hybrid-Option im Gateway  
**Commerce:** Genesys/Stockchain + Stripe  
**Datenhaltung:** Postgres + pgvector · Warehouse: TimescaleDB

Die technische Referenz zum FinX-Moda-Produktkonzept — Systemtopologie und Laufzeit-Flows für den konversationellen Commerce-Agenten, das Datenschema bis auf SQL-Ebene inklusive der finx-memory-Gedächtnisschichten, die AI-Schicht (Serving, Katalog-Pipeline, Training, Adapter, Versionierung), Checkout-Pfad, Deployment, Datenzonen und der Skalierungspfad.

---

## 01 · Systemüberblick

Alle Container, alle Verbindungen. Prototyp-Maschine: Docker Compose, hinter Caddy/Traefik mit TLS.

Zonen: **FINX-BB-ZONE** · **AI-ZONE (GPU)** · **MODA-ZONE**

| Container | Technologie | Verantwortung | Zustand |
|---|---|---|---|
| `moda-app` | Next.js (Node), PWA, shadcn/ui | Quiz, Swipe-Feed, Outfits, Merkliste; **rein präsentierend** | zustandslos |
| `moda-api` | Fastify (TS), Zod aus `finx-schemas` | Agent-Loop (Tool-Calling: Katalog-Suche, Live-Check, Fit-Check, Cart), Feed-Komposition, Event-Erfassung, Outbox, Session/Konto, Checkout-Orchestrierung | zustandslos (Zustand in `moda-db`) |
| `finx-memory` | Eigenes TS-Paket (in `moda-api` eingebettet) + Python-Teil in `moda-ml` | Lernschicht: L1-Graph-API, L2-Episoden (Schreiben/Retrieval), L3-Stilvektor-Updates, Lösch-Semantik über alle Schichten; Konsolidierung als `moda-ml`-Job. Teil der `finx-*`-Familie | Bibliothek — Zustand in `moda-db` |
| Commerce-Adapter | Modul in `moda-api` + `moda-ml` | Genesys/Stockchain: Live-Bestand/Preis-Checks vor Anzeige, Order-Erstellung, Katalog-Delta für die Pipeline | zustandslos |
| Stripe | Hosted Checkout (Phase 0) → Payment Element (Phase 1) | Zahlung; Moda hält keine Kartendaten — nur Order-Referenzen | extern |
| `moda-db` | Postgres 16 + pgvector (HNSW) | Operative Wahrheit: SKUs, Embeddings, Attribute, Nutzer, Events, Reco-Log | persistent, PII-Zone |
| `moda-ml` | Python (Polars, scikit/LightGBM), Cron-Jobs | Katalog-Sync + Pipeline, Ranker-Training, Offline-Evals | zustandslos, Jobs |
| Objektablage | MinIO (S3-API) oder Verzeichnis | SKU-Bilder (Original + Thumbnails) | persistent |
| `finx-models` | LiteLLM vor vLLM + TEI/Infinity | Ein OpenAI-kompatibler Endpoint für alles; Routing, Auth, `inference_log` | zustandslos |
| `finx-ingest` | Fastify (TS) | Einziger Schreibpfad ins Warehouse, Key je Produzent, idempotente Upserts | zustandslos |
| Warehouse | TimescaleDB (eigene DB, gleiche PG-Instanz möglich) | Pseudonymisierte Events, Metriken, `model_version`/`inference_log` | persistent, Pseudonym-Zone |

Weitere Akteure: Nutzer-Browser/PWA, Genesys/Stockchain Live-Commerce-API (Bestand · Preise · Versand · Checkout-Rechte), Stripe Hosted Checkout → Payment Element, OpenBB Workspace (Cloud) als Moda-Ops-App, `finx-backend` (X-API-KEY).

---

## 02 · Laufzeit-Flows

### Flow 0 — Konversations-Turn (der Agent-Pfad)

1. Client sendet Turn, z. B. *"Rooftop-Dinner Dubai, smart-casual, unter €600"*
2. `moda-api` Agent-Loop baut Kontext: `finx-memory` lädt L1-Graph + L2-Episoden per Ähnlichkeit
3. LLM-Turn über `finx-models` (System + L1/L2-Kontext + Tools)
4. Tool-Call `katalog_suche(anlass, budget, stil)` → pgvector-Kandidaten + Attribut-Filter + L3-Ranking in `moda-db`
5. Live-Check (Bestand, Preis) bei Genesys/Stockchain für Top-Treffer
6. Tool-Ergebnis → finaler Turn (Katalog-Layout + Begründungen)
7. Client erhält generierten Katalog-Abschnitt (**nur live verfügbare Teile**)
8. Turn asynchron als L2-Episode schreiben
9. Nachts: Konsolidierungs-Job destilliert Episoden zu L1-Fakten

### Flow A — Feed-/Katalog-Nachladen (LLM-frei)

Heißer Pfad bleibt LLM-frei. Ziel **p95 < 150 ms**.

1. `GET /feed` (Session-Token)
2. Stilvektor + Ausschlussliste laden
3. Kandidaten: Top-200 per HNSW-Ähnlichkeit
4. Re-Ranking: `Score = Ähnlichkeit + Popularität + Diversitätsstrafe + Explorations-Slot (ε)`
5. `RECO_LOG` schreiben (Kandidaten, Scores, Ranker-Version, Positionen)
6. Feed-Seite (24 SKUs, Bilder via CDN/Objektablage)

### Flow B — Event → Lernen → Warehouse

1. `POST /events` (like, sku, Position, Feed-Kontext)
2. Event speichern (Tx) + Outbox-Zeile
3. Stilvektor-Update: `v' = norm((1−α)·v + α·e_sku)`  
   like: α≈0.15 · skip: α≈−0.05 · save: α≈0.25
4. 204 an Client
5. Outbox-Drainer (alle 5 s, Batch) → `POST /ingest/moda/events` (`pseudo_id`, Batch, Event-IDs)
6. `finx-ingest`: idempotente Upserts (`event_id` als Konfliktschlüssel)
7. 200 → Outbox-Zeilen als versendet markieren

Synchron nur `moda-db`. Warehouse asynchron über die Outbox.

### Flow C — Katalog-Sync (nachts oder auf Webhook)

1. Delta seit letztem Sync (`updated_since`) von der SKU-API
2. Normalisieren + `content_hash` bilden
3. Hash unverändert → nur Preis/Verfügbarkeit aktualisieren
4. Neu/geändert → Bilder laden + Thumbnails; Bild-Embedding (SigLIP/FashionCLIP-Klasse); Vision-LLM: Attribute als JSON gegen festes Schema; Schema-Validierung, Konfidenz-Check, Retry; Upsert `sku` + `sku_embedding` + `sku_attributes`
5. Sync-Lauf protokollieren (Zahlen, Fehler)

Idempotent per SKU-ID + `content_hash`. Fehler je SKU isoliert.

---

## 03 · Datenarchitektur

Operative Wahrheit: `moda-db`. Migrationen via dbmate/Atlas (FinX-BB-Plan).

```sql
-- Katalog
CREATE TABLE sku (
  sku_id        text PRIMARY KEY,
  title         text NOT NULL,
  category      text NOT NULL,             -- top/bottom/shoes/layer/acc
  price_cents   integer NOT NULL,
  currency      text NOT NULL DEFAULT 'EUR',
  image_key     text NOT NULL,
  buy_url       text,
  available     boolean NOT NULL DEFAULT true,
  content_hash  text NOT NULL,
  source_meta   jsonb NOT NULL DEFAULT '{}',
  updated_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE sku_embedding (
  sku_id        text PRIMARY KEY REFERENCES sku ON DELETE CASCADE,
  embedding     vector(768) NOT NULL,
  model_version text NOT NULL
);
CREATE INDEX ON sku_embedding USING hnsw (embedding vector_cosine_ops);

CREATE TABLE sku_attributes (
  sku_id      text PRIMARY KEY REFERENCES sku ON DELETE CASCADE,
  silhouette  text, occasion text[], style text[],
  pattern     text, season text[],
  confidence  real,
  reviewed    boolean NOT NULL DEFAULT false,
  extracted_by text NOT NULL
);

-- Nutzer & Lernen
CREATE TABLE app_user (
  id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  pseudo_id  uuid UNIQUE NOT NULL DEFAULT gen_random_uuid(),
  email      text UNIQUE,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE user_profile (
  user_id      uuid PRIMARY KEY REFERENCES app_user ON DELETE CASCADE,
  style_vector vector(768) NOT NULL,
  prefs        jsonb NOT NULL DEFAULT '{}',
  updated_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE event (
  event_id   uuid PRIMARY KEY,
  user_id    uuid NOT NULL REFERENCES app_user ON DELETE CASCADE,
  session_id uuid NOT NULL,
  sku_id     text REFERENCES sku,
  type       text NOT NULL CHECK (type IN
    ('view','like','skip','save','click_out','outfit_accept','outfit_reject','quiz_choice')),
  position   smallint,
  ranker_ver text NOT NULL,
  context    jsonb NOT NULL DEFAULT '{}',
  ts         timestamptz NOT NULL DEFAULT now()
) PARTITION BY RANGE (ts);

CREATE TABLE reco_log (
  id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  session_id uuid NOT NULL,
  ranker_ver text NOT NULL,
  candidates jsonb NOT NULL,
  ts         timestamptz NOT NULL DEFAULT now()
) PARTITION BY RANGE (ts);

CREATE TABLE outbox (
  id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  payload    jsonb NOT NULL,               -- pseudo_id, nie email
  sent_at    timestamptz
);

-- L1: Strukturiertes Gedächtnis (Style/Fit-Graph)
CREATE TABLE user_fact (
  id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  user_id     uuid NOT NULL REFERENCES app_user ON DELETE CASCADE,
  kind        text NOT NULL,               -- brand_pref | silhouette | color_world | no_go |
                                           -- price_band | measurement | brand_size | occasion_need
  subject     text NOT NULL,
  value       jsonb NOT NULL,
  confidence  real NOT NULL DEFAULT 1.0,
  source      text NOT NULL,               -- stated | consolidated | derived_from_return
  updated_at  timestamptz NOT NULL DEFAULT now(),
  UNIQUE (user_id, kind, subject)
);

-- L2: Episodisches Gedächtnis (hash-gekettete Sessions)
CREATE TABLE episode (
  id          uuid PRIMARY KEY,
  user_id     uuid NOT NULL REFERENCES app_user ON DELETE CASCADE,
  seq         bigint NOT NULL,
  prev_hash   text NOT NULL,
  hash        text NOT NULL,
  summary     text NOT NULL,
  embedding   vector(768) NOT NULL,
  refs        jsonb NOT NULL DEFAULT '{}',
  prompt_sha  text,
  consolidated boolean NOT NULL DEFAULT false,
  ts          timestamptz NOT NULL DEFAULT now(),
  UNIQUE (user_id, seq)
);
CREATE INDEX ON episode USING hnsw (embedding vector_cosine_ops);

-- Commerce
CREATE TABLE purchase (
  id            uuid PRIMARY KEY,
  user_id       uuid NOT NULL REFERENCES app_user ON DELETE CASCADE,
  sku_id        text NOT NULL REFERENCES sku,
  merchant_order text,
  stripe_ref    text,
  size_bought   text,
  status        text NOT NULL CHECK (status IN ('placed','shipped','delivered','returned')),
  return_reason text,
  ts            timestamptz NOT NULL DEFAULT now()
);
```

### Wachstum & Aufbewahrung

- 10.000 SKUs ≈ 30 MB Vektoren (768 float32)
- Events: 1.000 aktive Nutzer × 200 Events/Monat ≈ 200k Zeilen/Monat — Monats-Partitionen
- `reco_log` wächst am schnellsten → Aufbewahrung 6–12 Monate, ältere Partitionen als Parquet exportieren, dann droppen

### Warehouse-Vertrag

**Über die Outbox:** Events unter `pseudo_id` (`event_id` als Idempotenz-Schlüssel), tägliche Katalog-Kennzahlen, Empfehlungs-Performance je `ranker_ver`. Session-Ausgänge in fünf Terminal-States: `completed` / `failed` / `abandoned_early` / `abandoned_mid` / `unreported_active`.

**Nie über die Grenze:** E-Mail, Konto-IDs, prefs-Inhalte, IPs.

Löschkaskade als Job: Konto löschen → `ON DELETE CASCADE` in `moda-db` → Lösch-Event mit `pseudo_id` an `finx-ingest` → Warehouse-Zeilen löschen, betroffene Parquet-Exporte neu schreiben.

---

## 04 · AI-Architektur

### Modell-Roster über das Gateway

| Rolle | Modellklasse | Serving | Wann | Latenz-Budget |
|---|---|---|---|---|
| Bild-Embedding | SigLIP/CLIP-Klasse, ggf. FashionCLIP | TEI/Infinity | Katalog-Sync (Batch) + Quiz-Bilder | Batch |
| Attribut-Extraktion | Vision-LLM ~7–8B, quantisiert (Qwen-VL-Klasse) | vLLM | Katalog-Sync (JSON-Schema-Modus) | Batch |
| Stylist / Begründungen | LLM ~8–14B, quantisiert (AWQ/GPTQ) | vLLM | Outfit-Ansicht (async nachgeladen) | < 2 s, cached je Outfit |
| Agent-Loop | Großes OSS self-hosted **oder** Frontier-API (Zero-Retention) für Phase 0 | vLLM bzw. Gateway-Route | Jeder Chat-Turn (Tool-Calling) | Erste Tokens < 2 s, gestreamt |
| Ranker (L3) | LightGBM/GBDT, später Two-Tower | ONNX/JSON in `moda-api`, kein GPU | Jeder Feed-Request | < 5 ms |
| Adapter (L4) | LoRA: Domäne → Kohorte → (später) Nutzer | vLLM LoRA-Loading, nur self-hosted | Fashion-Query, Outfit-Logik, Extraktion | wie Basis-Modell |
| Konsolidierung (L2→L1) | dasselbe LLM, Batch | vLLM, nachts | Episoden → Graph-Fakten | Batch |

VRAM-Budget Prototyp (24-GB-Karte): Vision-LLM 7–8B q ≈ 7–9 GB + Stylist 8B q ≈ 6–8 GB + Embeddings 1–2 GB + KV-Cache. v1-Vereinfachung: ein Vision-LLM für Extraktion + Stylist, erst bei Qualitätsbedarf trennen.

### Attribut-Vertrag (`finx-schemas`)

```json
{
  "silhouette": "oversized | fitted | straight | a-line | ...",
  "occasion":   ["casual","office","evening","sport"],
  "style":      ["minimal","street","classic","boho"],
  "pattern":    "solid | striped | floral | graphic | ...",
  "season":     ["spring","summer","autumn","winter"],
  "confidence": 0.0
}
```

Regeln: geschlossene Vokabulare, vLLM im JSON-Schema-Modus, `confidence < 0.6` → `reviewed=false`-Queue, Stichproben-Review als fester Pipeline-Schritt.

### Trainings- und Eval-Schleife (Stufe 2)

`event` + `reco_log` (mit Positionen, debiased) → `moda-ml` LightGBM-Ranker (Features: `cos(v_user, e_sku)`, Attribut-Matches, Preis-Fit, Popularität, Recency) → Offline-Eval Recall@k · NDCG gegen Holdout → `model_version` ins Warehouse → Deploy ONNX/JSON in `moda-api` → A/B neue Version vs. aktive (Kohorten).

Prompts sind versionierte Capabilities: Identität = `sha256(normalize(text))`. Jede Episode und jedes Extraktionsergebnis trägt den Prompt-Hash.

DLP vor der Extern-Route (Phase-0 Frontier-API): Gateway scannt Turn-Kontexte gegen Secret-/PII-Muster. Governance aus FinX-BB Abschnitt 05 gilt für jeden Moda-Inferenz-Call (allow/deny/cosign, Reserve/Reconcile, 4-Bucket-Metering, `prompt_sha256`, fail-closed, keyless).

Versionierung: Embedding-Modell, Extraktions-Modell und Ranker je eine Version. `sku_embedding.model_version` verhindert Mischvergleiche. Modellwechsel = kompletter Re-Embed + Stilvektoren neu aus jüngsten Likes.

Jeder Gateway-Call → `inference_log` im Warehouse.

**Ausfall der AI-Zone bricht Moda nicht ganz:** Feed bleibt LLM-frei (Flow A). Konversation degradiert auf strukturierte Suche. Outfit-Begründungen auf regelbasierte Texte. Katalog-Sync wartet. Gateway kann auf gehostete OSS-APIs routen, ohne dass `moda-api` sich ändert.

Lösch-Semantik ist Teil der AI-Schicht: Konto löschen verwirft L1+L2 (CASCADE), setzt L3 zurück, entfernt später per-User-Adapter, plus Kaskaden-Job über `pseudo_id` ins Warehouse.

---

## 05 · Deployment & Betrieb

Compose-Topologie auf der Prototyp-Maschine:

- `caddy` — TLS + Routing: `app.moda.x`, `api.moda.x`, `ingest.finx.x`
- `moda-app` — Node, Port intern
- `moda-api` — Fastify; ENV: `DB_URL`, `GATEWAY_URL`, `INGEST_KEY_MODA`
- `moda-ml` — Cron-Container (ofelia o. ä.): sync 03:00, training wöchentlich
- `postgres` — eine Instanz, drei DBs: `moda`, `warehouse`, `openbb`
- `minio` — SKU-Bilder (oder Bind-Mount)
- `litellm` — `finx-models` Gateway; Keys je Konsument
- `vllm` — GPU; `--gpu-memory-utilization` begrenzt
- `tei` — Embeddings
- `finx-ingest` — eigener Key je Produzent
- `uptime-kuma` — Heartbeats: sync-Lauf, Outbox-Lag, GPU-Health
- `backup` — pgBackRest/wal-g → Off-Site

Netz: Nur Caddy exponiert 443; alles andere intern. Workspace erreicht ausschließlich `finx-backend`-Routen (X-API-KEY + CORS auf `pro.openbb.co`). AI-Zone von außen unerreichbar.

Secrets: Env-Dateien außerhalb des Repos; ein Key je Produzent/Konsument, serverseitig gehasht.

Backups ab Tag 1 (FinX-BB Phase 0): pgBackRest Off-Site + getesteter Restore; MinIO-Bucket-Sync für Bilder.

Monitoring: Uptime-Kuma für Katalog-Sync, Outbox-Lag, GPU (`nvidia-smi`-Exporter), API-p95. Rest: Moda-Ops-Dashboard aus dem Warehouse.

---

## 06 · Datenzonen

| Zone | Inhalt |
|---|---|
| **Zone 1 · PII** (nur `moda-db`) | E-Mail · Konto · Sessions · `pseudo_id`-Mapping · finx-memory L1–L2: Graph, Maße, Größen, Episoden, Käufe |
| **Zone 2 · Pseudonym** (Warehouse) | Events + Reco-Performance unter `pseudo_id`; Katalog- & Modell-Metriken |
| **Zone 3 · Anonym** (Workspace-Widgets) | Aggregate: Raten, Kurven, Kohorten-Vergleiche |

Grenzen sind **Code-Grenzen**: Outbox-Payloads aus einem Zod-Schema ohne E-Mail-/Konto-Felder; Workspace-Widgets nur Aggregat-Queries des `finx-backend`. Zone 2 bleibt DSGVO-relevant. Löschkaskade endet erst in Zone 2.

---

## 07 · Skalierungspfad

| Auslöser | Änderung | Was gleich bleibt |
|---|---|---|
| GPU-Last steigt | Zweite Maschine in die AI-Zone; LiteLLM routet nach Modell | Gateway-URL, alle Aufrufer |
| DB-Last trennt sich | Warehouse auf eigene Instanz (Replikation, dann Umschalten) | Ingest-API als einziger Schreibpfad |
| Event-Volumen sprengt Outbox-Batching | Queue (NATS) zwischen Outbox und Ingest | Event-Schema, Idempotenz-Semantik |
| Feed-p95 kippt trotz Indexen | Kandidaten-Cache je Stil-Cluster; erst danach dedizierter Vektor-Dienst | `moda-api`-Schnittstelle |
| Ranker braucht mehr als GBDT | Two-Tower-Training in `moda-ml`, Serving weiter als exportiertes Modell in `moda-api` | Kein Online-GPU-Serving im Feed-Pfad |
| L4 wächst | Adapter-Registry + vLLM-Multi-LoRA; Training als `moda-ml`-Batch auf zweiter Maschine | Basis-Modell, Gateway-Route, `finx-memory`-API |
| Native App wird nötig | `moda-api` ist bereits die vollständige Schnittstelle | API-Verträge aus `finx-schemas` |

**Invarianten:** ein Schreibpfad ins Warehouse (`finx-ingest`), kein LLM im Feed-Pfad, Zonengrenzen als Code, jede Modellart versioniert.
