# Permit-to-Proof

**A verifiable LLM system that reads environmental permits, extracts every compliance obligation, and proves each one is grounded in the source before trusting it.**

Factories and industrial plants operate under environmental permits: long legal documents that set how much of each pollutant they may release, how often they must measure it, and when they must report. A single permit can hold hundreds of separate obligations, and reading them by hand is slow and error prone.

A language model can extract those obligations automatically. But models sometimes invent text, and in a compliance setting a fabricated or missed obligation is a real liability. So the model cannot be trusted on its own word.

Permit-to-Proof uses an LLM to extract the obligations, then uses a separate, deterministic checker to verify every one against the permit, confirming the exact supporting text genuinely exists. Obligations that pass are **Verified**; those that fail are **Flagged** or sent to a human review queue.

> **The rule the whole system rests on: the AI proposes, the deterministic checker disposes. No obligation is ever marked Verified on the model's say so alone.**

---

## Quickstart (one click)

You only need Python 3 installed (3.10 to 3.13 recommended).

**Windows:** double click `run.bat`.

**macOS / Linux:** run `./run.sh` (you may need `chmod +x run.sh` first).

The launcher creates a virtual environment, installs the pinned dependencies, and opens the app at `http://localhost:8501`. The first run takes a minute to install; later runs start in seconds. The bundled sample permit runs immediately, fully offline, with **no API key**, using the built in Mock backend.

### Manual setup (if you prefer)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/main.py
```

---

## Using the app

1. In the sidebar, pick a backend (Mock by default), set the routing threshold, and keep the verification layer ON.
2. Leave **Use bundled sample permit** checked, or uncheck it to upload your own permit PDF.
3. Click **Run extraction**.
4. Read the summary: total obligations, verified rate, how many were routed to a human, and the **error detection lift** the verification layer adds over trusting the raw model.
5. Expand any obligation to see its fields, its verification checks, and the exact supporting text with its match type. Ungrounded quotes (likely hallucinations) are called out in red.
6. Filter by status or search the text. Accept or reject any obligation; your decisions persist for the session.
7. Download everything as JSON or CSV, including statuses, checks, and match types.

---

## The verification layer (the heart of the project)

Stage 3 is pure, deterministic Python. It never calls a model and never touches the internet. It takes one candidate obligation plus the text of the segment it came from and returns the same obligation with its checks filled in. Seven checks run on every obligation, in a fixed order:

| Check | Confirms | Severity |
|-------|----------|----------|
| `schema_complete` | Required fields present and well typed (a numeric limit has value, unit, and operator) | error |
| `grounded` | The `source_quote` actually appears in the cited segment; records `match_type` as exact, fuzzy, or none | error |
| `citation_present` | A permit section or regulatory reference is recorded | warning |
| `unit_valid` | The unit fits the parameter type (air versus water): a domain check | warning |
| `range_plausible` | The value sits within a sensible envelope for that parameter and unit: a domain check | warning |
| `operator_consistent` | The operator matches the wording ("shall not exceed" implies a maximum): a domain check | warning |
| `no_duplicate` | Not a repeat of an obligation already extracted | info |

The three domain checks (`unit_valid`, `range_plausible`, `operator_consistent`) encode environmental compliance semantics that general purpose verifiers do not have. They are real, not placeholders, and they are the technical embodiment of the project's novelty claim.

**The anti hallucination core:** if the model returns `source_quote` text that does not appear in the cited segment, the obligation fails grounding (`match_type = none`) and is flagged, no matter how confident the model sounded. Matching is robust to spacing, case, and punctuation, but not so loose that invented text slips through.

### How status is decided

* Any failed **error** check (missing required field, or ungrounded quote) routes to **Flagged**.
* No errors, but a failed **warning** check or confidence below the threshold routes to **Needs review**.
* All checks pass and confidence is at or above the threshold routes to **Verified**.
* User overrides set **Accepted** or **Rejected** and are never silently dropped.

---

## Confidence and routing

Confidence is a single number in `[0, 1]` combining three signals (see `app/core/score.py`):

* the model's own self reported confidence,
* grounding strength (exact beats fuzzy beats none),
* the share of checks passed, weighted by severity.

The formula is a fixed weighted average, documented and unit tested, so identical inputs always give identical outputs. The routing threshold is a slider in the UI; sliding it trades automation against human review effort live. That slider **is** the trade off curve the research paper reports.

---

## Architecture

Five stages, kept separate so each can be tested alone. The verification layer never depends on the LLM.

```
PDF
  -> ingest()          List[Segment]    (text, page, char positions)
  -> extract()         List[Obligation] (raw, status = PENDING)      [LLM backend]
  -> verify()          List[Obligation] (checks[] + match_type)      [deterministic]
  -> score_and_route() List[Obligation] (confidence, status)         [deterministic]
  -> UI renders + user overrides + export
```

| Stage | Its job | It must not |
|-------|---------|-------------|
| 1. Ingest | Read the PDF into ordered text segments with page numbers and char positions | Call the LLM or interpret meaning |
| 2. Extract | An LLM backend turns segments into candidate obligation records | Decide verification status |
| 3. Verify | Deterministic checks on each candidate; attach reasons and match type | Call the LLM or change the source text |
| 4. Score and route | Compute confidence; assign status by thresholds | Hide why something was flagged |
| 5. Present | Browser UI: tables, badges, source text, overrides, metrics, export | Contain logic that belongs in the core |

---

## Backends (switchable at runtime)

One interface, three implementations. The pipeline depends only on the interface, so backends are interchangeable.

| Backend | Needs | Role |
|---------|-------|------|
| **Mock** (default) | Nothing: no key, no internet | Runs instantly on a clean machine; powers the reproducible tests. Does honest regex extraction from the permit text, so every quote is grounded. |
| **OpenAI** | `OPENAI_API_KEY` in `.env`; model name configurable | Real hosted extraction with structured JSON output. |
| **Ollama** | A local Ollama server; model name configurable | Local, private extraction for the cost and privacy comparison. |

Copy `.env.example` to `.env` and fill in a key to enable the real backends. Both real backends ask the model for JSON matching the obligation schema and insist the exact supporting text be copied into `source_quote`, so grounding stays meaningful. A bad model response becomes flagged data, never a crash.

### Verification ON / OFF (the headline experiment)

The pipeline has a flag (and the UI a checkbox) to run with the verification layer **disabled** (raw model output, trusted as is) versus **enabled**. The app always runs both and reports the **error detection lift**: how many obligations the verification layer surfaces that the raw pipeline would have passed silently. This is the in app proxy for the paper's headline result.

---

## Project structure

```
permit-to-proof/
|- run.bat / run.sh           one click launchers
|- evaluate.py                evaluation harness CLI (paper metrics + figures)
|- requirements.txt           pinned dependencies
|- pyproject.toml             project metadata + packaging
|- LICENSE / CITATION.cff     MIT license + citation metadata
|- pytest.ini                 test configuration
|- .env.example               documents OPENAI_API_KEY and OLLAMA_HOST
|- .streamlit/config.toml     headless server + theme settings
|- app/
|  |- main.py                 Streamlit UI (presentation only)
|  |- ui_theme.py             the futuristic HUD theme (presentation only)
|  |- pages/
|  |  |- 2_Evaluation.py      interactive evaluation page
|  |- core/
|  |  |- schema.py            Pydantic data contracts
|  |  |- ingest.py            PDF -> segments
|  |  |- verify.py            the deterministic verification layer
|  |  |- score.py             confidence + routing
|  |  |- pipeline.py          runs the stages; metrics
|  |- llm/
|  |  |- base.py              backend interface + defensive parsing
|  |  |- mock.py              offline default backend
|  |  |- openai_backend.py
|  |  |- ollama_backend.py
|  |- eval/
|     |- gold.py              gold answer-key loader (pydantic)
|     |- metrics.py           P/R/F1, verification lift, ECE, selective curve
|     |- report.py            metrics.json + report.md + figures
|- docs/
|  |- methodology.md          evaluation protocol + threats to validity
|  |- domain_knowledge.md     provenance of the unit/range checks
|- sample_data/
|  |- sample_permit.pdf       bundled synthetic permit
|  |- make_sample_permit.py   regenerates the sample
|  |- gold/sample_permit.json illustrative gold set (self-test only)
|- tests/
   |- conftest.py             shared fixtures
   |- test_verify.py          the heaviest test file
   |- test_schema.py
   |- test_score.py
   |- test_pipeline.py
   |- test_backends.py
   |- test_eval.py            evaluation harness tests
   |- test_app.py             Streamlit UI smoke tests (AppTest)
```

---

## Testing

```bash
pytest
```

The whole suite runs offline with no API key, in a few seconds. The verification layer is covered most heavily: crafted obligations (complete, missing fields, ungrounded quote, out of range value, wrong unit, wrong operator, duplicate) are fed in and the exact checks, match type, and final status are asserted, including adversarial grounding cases (fabricated value, unit swap, negation flip) that must NOT pass as fuzzy matches. The OpenAI and Ollama backends import safely; their live network calls are opt in and run only when you set `PTP_RUN_LIVE_BACKENDS=1`, so a stray local model server can never pull the default suite onto the network.

---

## Evaluation (reproducing the paper's metrics)

The artifact does not just run the workflow; it measures it. The evaluation harness (`app/eval`, `evaluate.py`) grades extraction against a gold answer key and produces exactly the four numbers the research paper reports (see [docs/methodology.md](docs/methodology.md)):

```bash
python evaluate.py --out eval_out          # bundled synthetic permit + illustrative gold
```

This writes `eval_out/metrics.json`, a `report.md`, and four figures (reliability diagram, automation-vs-accuracy, risk-coverage, error-detection lift), and prints a summary:

* **Extraction quality** — precision, recall, F1 against the gold set.
* **Verification lift** — error-detection recall with the deterministic layer ON vs OFF (the headline result), measured against gold rather than the in-app proxy.
* **Calibration** — Expected Calibration Error (ECE), MCE, and a reliability diagram, for both the calibrated and the raw model confidence.
* **Selective-prediction trade-off** — automation rate vs auto-accept accuracy, and the operating point (most automation at a target accuracy).

The same numbers are shown interactively in the **Evaluation** page of the browser app (the second item in the sidebar nav). You can also point the harness at a real permit and a real key: `python evaluate.py --pdf my_permit.pdf --gold my_key.json --backend OpenAI`.

**Honesty note.** The bundled gold set (`sample_data/gold/sample_permit.json`) is labeled `ILLUSTRATIVE_AUTHOR_KNOWN`: it is the known truth of the *synthetic* permit, used to self-test the harness end to end. It is **not** the expert answer key the paper requires, which must be built by a domain expert on real permits ([docs/methodology.md](docs/methodology.md), master document A5.4). Every figure and panel carries this provenance so the two are never confused.

---

## Documentation

* [docs/methodology.md](docs/methodology.md) — evaluation protocol, metric definitions, gold-set construction, threats to validity, and the Data and Code Availability statement.
* [docs/domain_knowledge.md](docs/domain_knowledge.md) — provenance and justification of the unit and range checks (Title V / NSPS / NPDES program structure), and what they do and do not claim.

---

## Data and reproducibility

* The bundled `sample_permit.pdf` is **entirely synthetic and fictional**, generated by `sample_data/make_sample_permit.py`. It carries no copyright and lets the repository run with zero downloads. It is deliberately rich enough to exercise every check, including a planted clause whose quote fails grounding so a Flagged row appears in the demo. The Mock backend also plants one obligation that duplicates another: because `no_duplicate` is an informational check, that row is surfaced in its check list and in the flag reason breakdown but, by design, remains Verified (info checks never change status).
* Real permits are uploaded by the user at runtime (public sources include EPA ECHO / NPDES and state Title V air permit portals).
* An illustrative gold set for the synthetic permit ships at `sample_data/gold/sample_permit.json` to self-test the evaluation harness. The expert labeled answer key for the paper's real measurement is a separate research asset, built by a domain expert on real permits (see the Evaluation section and [docs/methodology.md](docs/methodology.md)); it is needed only to measure accuracy, not to run the app.
* The Mock backend, the verification layer, and the scoring formula are all deterministic, so results and tests reproduce exactly.

---

## Out of scope

This is a local, single user tool. It does not log in users, deploy to the cloud, download permits automatically, or persist data between sessions beyond the file exports. Writing the research paper and building the labeled answer key are separate efforts.

---

## How this maps to the research

Permit-to-Proof is the working artifact behind a methods paper on verification reliability for LLM based compliance extraction. The artifact is built to produce exactly the numbers the paper needs: extraction quality, the error detection lift of verification ON versus OFF, per obligation confidence (with `match_type` logged for calibration), and a human review trade off curve driven by the routing threshold. The domain checks are what differentiate the contribution from general purpose, source grounded verifiers.

---

## License

Released for research and educational use. The bundled sample permit is fictional and is provided only to demonstrate the software.
