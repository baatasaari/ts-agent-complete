# LBG Targeted Support Agent Platform

**Version 2.0.0 · FCA PS25/22 live 6 April 2026 · GCP + ADK + Neo4j**

An enterprise-grade agentic AI platform that identifies the most appropriate
targeted support suggestion for an LBG consumer through a five-zone pipeline:
intent classification → trait graph construction → conversational gap-fill →
deterministic segment matching → PS25/22 compliance validation → audited delivery.

> **PS25/22 Note:** Mortgages, pure protection insurance, and debt/credit products
> are **explicitly out of scope** per PS25/22 Ch.3 / DC-001. This platform covers
> Retail Investments, Structured Deposits, DC Pension Accumulation, and
> DC Pension Decumulation only.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Architecture Overview](#architecture-overview)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Running the Interactive Demo](#running-the-interactive-demo)
6. [Running Each Zone Independently](#running-each-zone-independently)
7. [ML-Automatic Zone 2 System](#ml-automatic-zone-2-system)
8. [Running the Tests](#running-the-tests)
9. [Generating Test Data](#generating-test-data)
10. [Running the Regulatory Visualiser](#running-the-regulatory-visualiser)
11. [Zone-by-Zone Flow](#zone-by-zone-flow)
12. [PS25/22 Ontology Catalogue](#ps2522-ontology-catalogue)
13. [Compliance Check Framework](#compliance-check-framework)
14. [FCA Compliance Invariants](#fca-compliance-invariants)
15. [EAMGP Signal Taxonomy](#eamgp-signal-taxonomy)
16. [Known Limitations](#known-limitations)
17. [Production Deployment Notes](#production-deployment-notes)

---

## Project Structure

```
ts_agent/                                   ← project root
│
├── pyproject.toml                          ← build, pytest settings, coverage config
├── conftest.py                             ← pytest sys.path bootstrap
├── requirements-core.txt                   ← pinned core deps (demo + tests)
├── requirements-adk.txt                    ← google-adk (production LLM only)
├── run_demo.py                             ← interactive CLI demo (no LLM required)
├── generate_test_data.py                   ← generates 3 v2 CSV test datasets
│
├── ts_agent/                               ← source package
│   │
│   ├── config/
│   │   └── segments.py                    ← PS25/22 ontology: 14 situations,
│   │                                         14 segments, 14 suggestions, 43 checks
│   │
│   ├── domain/
│   │   └── models.py                      ← all domain types: TraitNode, TraitGraph,
│   │                                         SegmentHypothesis, ExplainabilityBundle,
│   │                                         CheckSeverity, CheckPhase, ComplianceCheckResult
│   │
│   ├── ml/
│   │   └── predictor.py                   ← sklearn pipeline, SegmentPredictor,
│   │                                         IterativeSegmentPredictor, SHAP, IG matrix
│   │                                         16 v2 feature columns (investment + pension)
│   │
│   ├── observability/
│   │   ├── signals.py                     ← EAMGP emit(), timed(), listener API
│   │   ├── session_store.py               ← SessionStore, SessionRecord, SessionIndex
│   │   └── session_builder.py             ← replays scenarios → SessionStore (visualiser)
│   │
│   ├── resilience/
│   │   └── graph_writer.py                ← ResilientNeo4jWriter: circuit breaker, DLQ
│   │
│   ├── explainability/
│   │   └── explainer.py                   ← ConsumerExplainer, PS25/22 mandatory
│   │                                         disclosures (DEL-001–DEL-008, DEL-014)
│   │
│   ├── zones/
│   │   ├── zone1_graph_builder.py         ← Zone 1: bank data → TraitGraph
│   │   ├── session_resume.py              ← Zone 2: disconnection recovery (INV-09)
│   │   ├── zone2/
│   │   │   ├── tools.py                   ← Zone 2: 4 ADK FunctionTools for gap-fill
│   │   │   │                                 + excluding characteristic enforcement (PDC-001)
│   │   │   └── agent.py                   ← Zone 2: ADK LlmAgent (Gemini 2.0 Flash)
│   │   ├── zone3/
│   │   │   ├── suggestion_engine.py       ← Zone 3: PS25/22 compliance check evaluation
│   │   │   │                                 (PDC + DEL phase checks, 8 core rules)
│   │   │   └── delivery_agent.py          ← Zone 4: DeliveryCoordinator, INV-05 gate
│   │   └── agent/
│   │       └── lead_agent.py              ← LeadOrchestrator: wires all zones
│   │
│   └── visualiser/                        ← regulatory audit visualiser
│       ├── app.py                         ← Dash app (localhost:8050)
│       ├── data_adapter.py                ← enriches signals; 40 v2 char_ids mapped
│       ├── static_report.py               ← standalone HTML report per session
│       └── components/                    ← six Plotly panel builders
│
└── tests/
    ├── fixtures/factories.py              ← all domain object factories
    ├── datasets/scenario_catalogue.py     ← 22 v2 canonical scenarios
    │                                         (14 EMIT · 2 REVIEW · 6 SUPPRESS)
    ├── evaluation/
    │   └── test_pipeline_evaluation.py    ← 113 E2E assertions (all 22 scenarios)
    ├── integration/
    │   └── test_end_to_end_csv.py         ← 79 integration tests driven by CSVs
    └── unit/                              ← 493 unit tests across 11 files
```

**Test totals:** 685 passed · 5 skipped · 1 xfailed · 0 failed · 36s

---

## Architecture Overview

```
═══════════════════════════════════════════════════════════════════════
 Configuration Layer (YAML-Driven)  —  All PS25/22 Data Externalized
═══════════════════════════════════════════════════════════════════════
    config/fca_ts_situations.yaml        (27KB)  ← 14 situations (COBS 9B.3)
    config/fca_ts_segmentations.yaml     (45KB)  ← Consumer segments
    config/fca_ts_suggestions.yaml       (54KB)  ← Ready-made suggestions
    config/fca_ts_compliance_checks.yaml (49KB)  ← PS25/22 rules
                                ▼
                     ts_agent.config.config_loader
                     (Loads all regulatory data from YAML)
───────────────────────────────────────────────────────────────────────

Mobile / Web Client
        │  JWT · rate-limit · routing
        ▼
    Apigee Gateway
        │
        ▼
Lead Orchestrator  (GCP Agent Engine)
        │  ← Uses ConfigLoader for all FCA PS25/22 data
        │
        ├─ Zone 0  ─ DeBERTa Intent Classifier ──────────── ClassifiedIntent
        │              14 intents → 14 situations (from fca_ts_situations.yaml)
        │
        ├─ Zone 1  ─ TraitGraphBuilder ─── bank data ─────── TraitGraph → Neo4j
        │              40 v2 char_ids across 6 branches
        │
        ├─ Zone 1.5 ─ IterativeSegmentPredictor  (sklearn)
        │              16-feature v2 vector · SHAP attribution · IG fill order
        │              (Uses fca_ts_segmentations.yaml for segment definitions)
        │
        ├─ Zone 2  ─ GapFillAgent  (ADK LlmAgent · Gemini 2.0 Flash)
        │              ├── get_next_question   (IG-ordered gap fill)
        │              ├── record_consumer_answer
        │              ├── check_graph_completeness
        │              └── match_segment ──── excluding char enforcement ── segment_id
        │                                      (PDC-001 · PS25/22 para 3.49)
        │                                      (from fca_ts_segmentations.yaml)
        │
        ├─ Zone 3  ─ SuggestionEngine ─── 8 core rules + PDC/DEL checks ── gate_disposition
        │              EMIT · HUMAN_REVIEW · SUPPRESS
        │              (Uses fca_ts_suggestions.yaml + fca_ts_compliance_checks.yaml)
        │
        └─ Zone 4  ─ DeliveryCoordinator
                       ├── Jinja2 template: TARGETED SUPPORT label (DEL-001)
                       │   nature-and-limitations (DEL-002) · MoneyHelper (DEL-006)
                       │   capital-at-risk (DEL-005) · no-consolidation (DEL-014)
                       │   (from fca_ts_suggestions.yaml)
                       ├── ExplainabilityBundle → Cloud Spanner (INV-05)
                       └── consumer_message released after write confirmed

Knowledge Layer:  Neo4j Causal Cluster  (europe-west2)
Observability:    EAMGP → structlog + OpenTelemetry → Cloud Operations
Visualiser:       Dash + Plotly  (localhost:8050  or  standalone HTML)
```

**YAML-Driven Architecture Benefits:**
- ✅ All FCA PS25/22 regulatory data externalized (178KB)
- ✅ Non-developer updates via YAML (no code changes)
- ✅ Version controlled regulatory changes
- ✅ Environment-specific configurations
- ✅ Easy audit trail of compliance updates

---

## Prerequisites

| Requirement | Minimum | Check |
|-------------|---------|-------|
| Python | 3.12 | `python3 --version` |
| uv | any | `uv --version` |

---

## Installation

### macOS / Linux

```bash
# Install uv (skip if installed)
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.zshrc

# Clone / enter project
cd ts_agent

# Install all dependencies (using pyproject.toml)
uv sync

# Activate virtual environment
source .venv/bin/activate

# Install visualiser dependencies
uv pip install plotly==6.7.0 dash==4.1.0 dash-bootstrap-components==2.0.4

# Verify
python -c "from ts_agent.config.segments import SEGMENTS; print(len(SEGMENTS), 'segments')"
# → 14 segments
```

### Windows

```batch
uv python install 3.12
cd ts_agent
uv sync
.venv\Scripts\activate.bat
uv pip install plotly==6.7.0 dash==4.1.0 dash-bootstrap-components==2.0.4
```

### Production ADK (GCP access required)

```bash
uv sync --extra adk
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=your-gcp-project-id
```

---

## Running the Interactive Demo

No LLM, no Neo4j, no cloud account required.

```bash
python run_demo.py
```

Select a scenario from the menu (1–8). The demo walks through Zone 1 → Zone 2
gap-fill (you answer questions in the terminal) → Zone 3 rule evaluation →
Zone 4 delivery with all PS25/22 mandatory disclosures.

**Trigger specific outcomes:**

| Outcome | How |
|---------|-----|
| HUMAN_REVIEW | Answer `yes` to the vulnerability question on any scenario |
| SUPPRESS (low confidence) | Enter very minimal answers (only 1–2 traits) |
| SUPPRESS (existing product) | Scenario 1: answer `yes` to "already hold S&S ISA?" |
| SUPPRESS (lump sum > £75k) | Scenario 4: enter `80000` as lump sum amount |

---

## Running Each Zone Independently

Each zone can be tested and run independently for development, debugging, and experimentation.

### Zone 1: TraitGraph Builder

**Quick Test** - Run in Python REPL:

```bash
python -c "
from ts_agent.zones.zone1_graph_builder import TraitGraphBuilder

builder = TraitGraphBuilder()
graph = builder.build_graph(
    party_ref='CUST-12345',
    intent_id='INTENT-CASH-DRAG',
    situation_id='SIT-INV-001',
    bank_data={
        'savings_balance': 15000,
        'monthly_surplus': 500,
        'account_tenure_months': 18,
    }
)

print(f'Built graph with {len(graph.nodes)} nodes')
print(f'Known: {len(list(graph.known_nodes()))}')
print(f'Missing: {len(list(graph.missing_nodes()))}')
"
```

### Zone 1.5: ML Segment Predictor

**Quick Test** - Run prediction:

```bash
python -c "
from ts_agent.ml.predictor import SegmentPredictor

predictor = SegmentPredictor()
hypothesis = predictor.predict(
    traits={
        'CHAR-F2A-I1': 500,   # monthly_surplus
        'CHAR-F2B-I1': 15000, # cash_deposits
        'CHAR-F2I-I1': False, # current_investments
    },
    situation_id='SIT-INV-002'
)

print(f'Predicted: {hypothesis.top_segment_id}')
print(f'Confidence: {hypothesis.top_confidence:.1%}')
print(f'SHAP top features: {hypothesis.shap_features[:3]}')
"
```

### Zone 2: Gap-Fill Agent with Real Gemini LLM

**🚀 NEW: Run Zone 2 with Real Gemini** (no mocking!)

This script runs the complete pipeline with **real Gemini LLM** generating questions:

```bash
# 1. Setup (one-time)
uv sync --extra adk
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=your-gcp-project-id

# 2. Run interactive conversation with Gemini
python run_zone2_with_gemini.py
```

**What happens:**
- Zone 1 builds TraitGraph from bank data
- Zone 1.5 runs baseline ML prediction
- **Zone 2 starts REAL Gemini conversation**
  - Gemini asks questions in natural language
  - You type answers interactively
  - ML predicts automatically after each answer
  - Shows confidence evolving in real-time
  - Stops when confidence reaches 75%

**Example session:**

```
Select a scenario:
1. First-time Investor (SIT-INV-002)
2. ISA Allowance (SIT-INV-003)
3. Lump Sum Investment (SIT-INV-004)

Enter choice (1-3): 1

🔧 Zone 1: Building TraitGraph from bank data...
   Built graph:
   - Total nodes: 12
   - Known: 4
   - Missing: 8

🤖 Zone 1.5: Running baseline ML prediction...
   Baseline prediction:
   - Top segment: SEG-INV-002
   - Confidence: 45.2%
   - Fill order: CHAR-F2I-I1, CHAR-P2A-I1, CHAR-P2B-I1...

💬 Zone 2: Starting Gemini conversation...
Press Enter to start the conversation with Gemini...

[Turn 1] Processing with Gemini...
🤖 Gemini: Hello! To help find the right support for you, 
           may I ask: Do you currently have any investments?

👤 Your answer: No

[Turn 2] Processing with Gemini...
   ℹ️  ML updated prediction automatically

🤖 Gemini: Thank you. What is your current age?

👤 Your answer: 28

[Turn 3] Processing with Gemini...
   ℹ️  ML updated prediction automatically

🤖 Gemini: Great! One more question - roughly how much 
           do you have left each month after expenses?

👤 Your answer: £400

  ✅ CONVERSATION COMPLETE!

Matched Segment: SEG-INV-002
Total turns: 3

📈 ML Confidence Evolution:
   Turn 1: SEG-INV-002 - 45.2% (Δ+45.2%)
   Turn 2: SEG-INV-002 - 68.7% (Δ+23.5%)
   Turn 3: SEG-INV-002 - 83.4% (Δ+14.7%)
```

**Key Features:**
- ✅ Real Gemini 2.0 Flash LLM (not mocked)
- ✅ Automatic ML prediction after every answer
- ✅ Config-based question prioritization
- ✅ Early stopping at 75% confidence
- ✅ Shows ML confidence evolution
- ✅ Complete audit trail logged

### Zone 2: Tools API (for custom integrations)

If you want to integrate Zone 2 tools into your own application:

```python
from ts_agent.zones.zone2.tools_ml_auto import (
    get_next_question_ml_prioritized,
    record_consumer_answer_ml_auto,
    check_graph_completeness_ml_aware,
    match_segment_ml_auto,
)

# These tools automatically run ML prediction
# See ZONE2_INTELLIGENT_MODE.md for full API documentation
```

### Zone 3: Suggestion Engine

**Quick Test** - Evaluate compliance:

```bash
python -c "
from ts_agent.zones.zone3.suggestion_engine import SuggestionEngine
from ts_agent.zones.zone1_graph_builder import TraitGraphBuilder
from ts_agent.ml.predictor import SegmentPredictor
from ts_agent.domain.models import ExplainabilityBundle

# Build minimal graph
builder = TraitGraphBuilder()
graph = builder.build_graph(
    party_ref='CUST-001',
    intent_id='INTENT-FIRST-TIME-INVESTOR',
    situation_id='SIT-INV-002',
    bank_data={'age': 28, 'savings_balance': 8000}
)

# Get prediction
predictor = SegmentPredictor()
hypothesis = predictor.predict(
    traits={'CHAR-P2A-I1': 28, 'CHAR-F2B-I1': 8000},
    situation_id='SIT-INV-002'
)

# Evaluate compliance
engine = SuggestionEngine()
result = engine.evaluate(
    segment_id='SEG-INV-002',
    graph=graph,
    hypothesis=hypothesis,
    bundle=ExplainabilityBundle()
)

print(f'Gate disposition: {result.gate_disposition.value}')
print(f'Suggestion: {result.suggestion_id}')
passed = len([c for c in result.checks if c.outcome == 'PASS'])
print(f'Checks passed: {passed}/{len(result.checks)}')
"
```

### Zone 4: Delivery Coordinator

**Quick Test** - Generate consumer message:

```bash
python -c "
from ts_agent.zones.zone3.delivery_agent import DeliveryCoordinator
from ts_agent.zones.zone3.suggestion_engine import SuggestionEngine
from ts_agent.zones.zone1_graph_builder import TraitGraphBuilder
from ts_agent.ml.predictor import SegmentPredictor
from ts_agent.domain.models import ExplainabilityBundle

# Build components (simplified)
builder = TraitGraphBuilder()
graph = builder.build_graph(
    party_ref='CUST-001',
    intent_id='INTENT-FIRST-TIME-INVESTOR',
    situation_id='SIT-INV-002',
    bank_data={'age': 28, 'savings_balance': 8000}
)

predictor = SegmentPredictor()
hypothesis = predictor.predict(
    traits={'CHAR-P2A-I1': 28, 'CHAR-F2B-I1': 8000},
    situation_id='SIT-INV-002'
)

engine = SuggestionEngine()
suggestion_result = engine.evaluate(
    segment_id='SEG-INV-002',
    graph=graph,
    hypothesis=hypothesis,
    bundle=ExplainabilityBundle()
)

# Generate delivery
coordinator = DeliveryCoordinator()
delivery = coordinator.deliver(
    suggestion_result=suggestion_result,
    bundle=ExplainabilityBundle()
)

print(f'Gate: {delivery.gate_disposition.value}')
if delivery.consumer_message:
    print(f'Message preview: {delivery.consumer_message[:200]}...')
print(f'Audit confirmed: {delivery.audit_confirmed}')
"
```

### All Zones Together: Full Pipeline Demo

Run the complete interactive demo (no LLM required):

```bash
python run_demo.py
```

This walks through all zones with keyboard input instead of Gemini.

---

## ML-Automatic Zone 2 System

**NEW:** Zone 2 now supports ML-automatic mode where prediction happens continuously
and automatically after each consumer answer, without requiring explicit LLM tool calls.

### Key Features

1. **Automatic Prediction**: ML runs after EVERY answer, not just at the end
2. **Config-Based Priorities**: Segment-specific trait priorities from configuration file
3. **4-Tier Prioritization**: SHAP → Segment Config → Global → Standard
4. **Full Explainability**: Prediction history, confidence delta, SHAP features logged
5. **Early Stopping**: Stops at 75% ML confidence (43% fewer questions)

### Architecture Flow

```
Zone 1: Build TraitGraph from bank data
   ↓
Zone 1.5: BASELINE prediction (turn=0)
   ↓ [Initial hypothesis + fill order]
Zone 2: Gemini conversation loop
   ├─ Ask question (prioritized by config + SHAP)
   ├─ Record answer
   ├─ → AUTOMATIC ML prediction (iterative refinement)
   ├─ Check confidence (75% threshold)
   └─ Repeat or finalize
   ↓
Zone 3: Suggestion engine
```

### Configuration File

Segment-specific priorities in `ts_agent/config/segment_fill_priorities.py`:

```python
SEG_INV_002_PRIORITY = {
    "priority_traits": [
        "CHAR-F2I-I1",  # Current Investments (key discriminator)
        "CHAR-F2A-I1",  # Monthly Surplus
        "CHAR-F2B-I1",  # Cash/Deposits
        "CHAR-P2A-I1",  # Age
        "CHAR-P2B-I1",  # Risk Tolerance
    ],
    "rationale": "First-time Investor: Key is no existing investments",
    "min_required": 3,
}
```

### Usage

```python
from ts_agent.zones.zone2.agent_ml_auto import ML_AUTO_GAP_FILL_AGENT
from ts_agent.zones.zone2.tools_ml_auto import (
    record_consumer_answer_ml_auto,
    get_next_question_ml_prioritized,
    check_graph_completeness_ml_aware,
)

# Use ML-automatic agent
agent = ML_AUTO_GAP_FILL_AGENT

# Or use tools directly
result = await record_consumer_answer_ml_auto(
    char_id="CHAR-F2A-I1",
    value="500",
    tool_context=mock_context
)

# ML prediction happens automatically!
print(f"ML prediction: {result['ml_prediction']}")
# {
#   "segment_id": "SEG-INV-002",
#   "confidence": 0.68,
#   "confidence_delta": +0.23,  # Improved by 23%
#   "high_confidence": False
# }

print(f"Next question: {result['next_char_id']}")  # From config priorities
print(f"Ready: {result['ready_for_match']}")
```

### System Prompt

The LLM agent uses a friendly yet expert conversational tone:

```
You are the LBG Targeted Support conversational assistant.

Your job is simple: collect financial information from consumers by asking
ONE question at a time. The system handles all ML prediction and question
prioritization automatically - you just focus on the conversation.

=== CONVERSATION RULES ===

1. **ONE question per turn** - never ask two things at once
2. **Natural language** - avoid jargon, be conversational
3. **Acknowledge answers** - show you're listening
4. **Handle uncertainty** - if consumer says "I don't know", record as "unknown"
5. **No product mentions** - only gather information, don't suggest products yet
6. **Privacy reminder** - occasionally mention data is only for finding suitable support
7. **Stop when ready** - trust the `ready_for_match` signal
```

### Benefits

- **43% fewer questions**: 3-6 questions instead of 8-10
- **Adaptive**: Questions change based on ML confidence
- **Transparent**: Full audit trail for regulators
- **Consumer-friendly**: Natural conversation, not interrogation
- **Configurable**: Easy to adjust priorities per segment

### Testing

```bash
# Unit tests (8 tests)
pytest tests/unit/zones/zone2/test_tools_ml_auto.py -v

# Integration tests (4 tests)
pytest tests/integration/test_ml_auto_zone2.py -v
```

### Documentation

See `ZONE2_INTELLIGENT_MODE.md` for complete technical documentation.

---

## Running the Tests

```bash
# Fast run — no coverage
pytest --no-cov -q
# → 685 passed · 5 skipped · 1 xfailed in ~36s

# With coverage (91%)
pytest
# → 685 passed · 5 skipped · 1 xfailed in ~46s

# Specific suites
pytest tests/unit/zones/zone3/     --no-cov -v   # 100 tests · engine + prohibitions
pytest tests/evaluation/            --no-cov -v   # 113 tests · E2E all 22 scenarios
pytest tests/integration/           --no-cov -v   # 79 tests  · CSV-driven (requires data)
pytest tests/unit/observability/    --no-cov -v   # 111 tests · signals + session builder

# Run with JUnit XML (for CI)
pytest --no-cov --junit-xml=test_report.xml
```

**Test result meanings:**

| Result | Count | Meaning |
|--------|-------|---------|
| PASSED | 685 | Assertion passed |
| SKIPPED | 5 | `zone2_suppress` scenarios: exclude from Zone-3-only tests (correct — verified in evaluation suite) |
| XFAILED | 1 | `test_paraphrase_variants_coerce_consistently`: descriptive paraphrases coerce to `str` vs numeric `int` — expected until Zone 0 entity extraction is implemented |

---

## Generating Test Data

```bash
python generate_test_data.py
# ~30 seconds; creates three files in the project root
```

| File | Rows | Purpose |
|------|------|---------|
| `ts_agent_consumer_profiles.csv` | 3,000 | E2E system testing. All 40 v2 char_id columns, all 12 rule outcomes, `gate_disposition` label. 14 segments × ~214 records. |
| `ts_agent_ml_training.csv` | 3,000 | ML training. 16 feature columns = `ALL_FEATURE_COLUMNS`. Pre-split 80/10/10 TRAIN/VAL/TEST. Domain-specific features are NaN for segments in other domains. |
| `ts_agent_sample_prompts.csv` | 44 | Consumer utterances. 14 intent prompts (one per situation) + 25 v2 trait gap-fill prompts + 5 edge cases. Each has 3 paraphrase variants. |

Then run the integration tests:

```bash
pytest tests/integration/ --no-cov -v   # 79 passing + 1 xfailed
```

---

## Running the Regulatory Visualiser

**Modern Scottish Widows / LBG branded interface** featuring professional teal and 
cyan colors, clean typography (Inter font), gradient backgrounds, and enhanced 
card shadows for a sophisticated compliance dashboard experience.

```bash
# Live Dash app
python -c "
from ts_agent.visualiser.app import create_app, build_demo_store
store = build_demo_store()
app   = create_app(store)
print(f'Loaded {store.record_count()} sessions')
app.run(debug=False, port=8050)
"
# Open http://localhost:8050
```

```bash
# Standalone HTML reports (no server needed)
python -c "
from ts_agent.visualiser.app import build_demo_store
from ts_agent.visualiser.static_report import generate_static_report
store = build_demo_store()
for gate in ['EMIT', 'HUMAN_REVIEW', 'SUPPRESS']:
    rec  = next(r for r in store.all_records() if r.gate_disposition == gate)
    path = generate_static_report(rec, f'ts_report_{gate.lower()}.html')
    print('Written:', path)
"
```

The visualiser has six layers: Session Overview (L2) · Conversation Transcript (L1)
· EAMGP Signal Trace (L3) · Compliance Gate Table (L4) · ML Prediction Chain (L5)
· Decision Flow Sankey (L6).

**Design highlights:**
- Scottish Widows teal (#006B6E) and LBG cyan (#009CA6) brand colors
- Gradient sidebar with modern Inter font (400-800 weights)
- Clean white cards with subtle shadows and teal accent borders
- Prominent warning banner for FCA confidentiality notice
- Responsive layout optimized for regulatory audit workflows

---

## Zone-by-Zone Flow

```
Consumer: "I want to make the most of my ISA allowance before April"
          │
          ▼  Zone 0 — DeBERTa (upstream)
          │  intent_id = "INTENT-ISA-ALLOWANCE"  →  SIT-INV-003
          │
          ▼  Zone 1 — TraitGraphBuilder
          │  Bank data: savings=£2,000 · account_tenure=24mo · no ISA sub this year
          │  tax_year_window=True (within 90 days of 5 April)
          │  Missing: CHAR-B3A-I1 (risk appetite) · CHAR-B3B-I1 (investment experience)
          │  Written to Neo4j (HARD gate — INV-01)
          │
          ▼  Zone 1.5 — IterativeSegmentPredictor
          │  P(SEG-INV-003 | known) = 0.79 · fill_order = [B3A, B3B]
          │
          ▼  Zone 2 — GapFillAgent  (Gemini 2.0 Flash)
          │  Turn 1: "On a scale of 1–5, how comfortable are you with investment risk?"
          │  Consumer: "About a 3"  →  CHAR-B3A-I1 = 3.0
          │  Turn 2: "Have you invested before?"
          │  Consumer: "No"  →  CHAR-B3B-I1 = 0
          │  match_segment: SEG-INV-003 ✓ (all including criteria met, no excluding chars)
          │
          ▼  Zone 3 — SuggestionEngine
          │  Candidate: SUG-INV-003 (STOCKS_SHARES_ISA_TAX_YEAR)
          │  R-001: segment match ✓ · R-002: age >= 18 ✓ · R-003: not vulnerable ✓
          │  R-004: no ISA sub this year ✓ · R-005: no risk floor for TAX_YEAR ✓
          │  R-006: no experience required for entry-level TAX_YEAR prompt ✓
          │  R-007: surplus positive ✓ · R-009: confidence 0.79 ≥ 0.75 ✓
          │  Gate: EMIT
          │
          ▼  Zone 4 — DeliveryCoordinator
          │  INV-05: ExplainabilityBundle → Spanner (write confirmed)
          │  Template renders:
          │    ⚠ TARGETED SUPPORT — This is targeted support, not personalised advice.
          │    "Use Your ISA Allowance Before Tax Year End"
          │    Characteristics: savings ≥ £500 · no ISA sub this year · within 90 days
          │    Not a comprehensive individual assessment.
          │    MoneyHelper: moneyhelper.org.uk
          │    Reference: <audit_id> | FCA firm: FRN-119278
          │
          Consumer sees the message.
```

---

## PS25/22 Ontology Catalogue

All defined in `ts_agent/config/segments.py`. **14 situations · 14 segments · 14 suggestions.**

### Domains

| Domain | Situations | Segments |
|--------|-----------|---------|
| Retail Investments | SIT-INV-001–006 | SEG-INV-001–006 |
| Structured Deposits | SIT-SD-001 | SEG-SD-001 |
| DC Pension Accumulation | SIT-PEN-001–003 | SEG-PEN-001–003 |
| DC Pension Decumulation | SIT-DEC-001–004 | SEG-DEC-001–004 |

### Situations (14)

| ID | Label | Trigger |
|----|-------|---------|
| SIT-INV-001 | Cash Drag — Under-invested Saver | Firm initiative |
| SIT-INV-002 | First-Time Investor — Knowledge Barrier | Consumer request |
| SIT-INV-003 | ISA Allowance Non-Utilisation — Tax Year End | Firm initiative |
| SIT-INV-004 | Lump Sum Capital Event — Investment Direction | Both |
| SIT-INV-005 | Dormant Investment Account — Re-engagement | Firm initiative |
| SIT-INV-006 | Regular Saver Seeking Investment Upgrade | Firm initiative |
| SIT-SD-001 | Maturing Fixed-Rate Deposit — Reinvestment Direction | Firm initiative |
| SIT-PEN-001 | Under-saving for Retirement — Below Adequacy Benchmark | Firm initiative |
| SIT-PEN-002 | Default Fund Disengagement — Lifecycle Mismatch | Firm initiative |
| SIT-PEN-003 | Life Event — Contribution Review Opportunity | Firm initiative |
| SIT-DEC-001 | Approaching Retirement — No Decumulation Plan (45–65) | Firm initiative |
| SIT-DEC-002 | Small Pot — Imminent Retirement Decision | Both |
| SIT-DEC-003 | Annuity Consideration — Guaranteed Income Exploration | Consumer request |
| SIT-DEC-004 | Drawdown Review — Consumer Already in Drawdown | Firm initiative |

### Segments (14) — Key Criteria

| Segment | Situation | Key Including Characteristics | Key Excluding Characteristics |
|---------|-----------|------------------------------|-------------------------------|
| SEG-INV-001 | SIT-INV-001 | savings £10k–£100k · no investment · tenure ≥ 12mo | Vulnerability · high-cost debt · HNW |
| SEG-INV-002 | SIT-INV-002 | aged 18–40 · employed/SE · savings £500–£25k · no investment | Vulnerability · active high-cost debt |
| SEG-INV-003 | SIT-INV-003 | no ISA sub this year · savings ≥ £500 · within 90 days of 5 April | Vulnerability · ISA limit already reached |
| SEG-INV-004 | SIT-INV-004 | lump sum £5k–£75k · no investment instruction | Vulnerability · lump sum > £75k (exit TS) · arrears |
| SEG-INV-005 | SIT-INV-005 | holds investment · inactive ≥ 12 months | Vulnerability · recently engaged |
| SEG-INV-006 | SIT-INV-006 | regular saving ≥ £50/mo for ≥ 6 months · no investment | Vulnerability · high-cost debt |
| SEG-SD-001 | SIT-SD-001 | deposit maturing ≤ 60 days · no reinvestment instruction | Vulnerability · deposit > £100k |
| SEG-PEN-001 | SIT-PEN-001 | aged 25–57 · contribution ≤ 8% · active DC · projected shortfall | Vulnerability · hardship · within 5 yrs of retirement |
| SEG-PEN-002 | SIT-PEN-002 | 100% default fund · no active selection ever · active DC | Vulnerability · recently engaged |
| SEG-PEN-003 | SIT-PEN-003 | life event signal · contribution ≤ 12% · active DC | Vulnerability · hardship · within 5 yrs of retirement |
| SEG-DEC-001 | SIT-DEC-001 | aged 45–65 · DC pension · no access plan | Vulnerability · already in drawdown · DB transfer |
| SEG-DEC-002 | SIT-DEC-002 | pot £5k–£30k · within 5 yrs of retirement · no access plan | Vulnerability · trivial commutation (<£2k) · above ceiling (>£30k) |
| SEG-DEC-003 | SIT-DEC-003 | expressed annuity interest · DC pension · aged ≥ 50 | Vulnerability · DB safeguarded benefits |
| SEG-DEC-004 | SIT-DEC-004 | in active drawdown · no review ≥ 18 months · cash drag flag | Vulnerability · recent independent advice |

### Suggestions (14) — Absolute Prohibitions

All suggestions carry `DC-002` (no pension consolidation) and `DC-003` (no specific annuity product). Pension domain suggestions additionally carry `DEL-014` (no-consolidation affirmative statement) and `DEL-006` (Pension Wise mandatory signpost, COBS 19).

---

## Compliance Check Framework

43 checks across 5 phases (fca_ts_compliance_checks.yml).

| Phase | IDs | Count | When Evaluated |
|-------|-----|-------|---------------|
| Pre-launch | PL-001–002 | 2 | Firm-level (not per session) |
| Design | DC-001–011 | 11 | Segment/suggestion design time |
| Pre-delivery | PDC-001–007 | 7 | Per consumer session (Zone 3) |
| Delivery | DEL-001–014 | 14 | Per delivery (Zone 4) |
| Monitoring | MON-001–009 | 9 | Ongoing (not per session) |

**Gate disposition mapping:**

| Severity | Gate outcome |
|----------|-------------|
| `HARD_BLOCK` | SUPPRESS (absolute halt) |
| `BOUNDARY_CHECK` | SUPPRESS (exit TS) |
| `SOFT_WARNING` | HUMAN_REVIEW |
| `DISCLOSURE_REQUIRED` (missing) | HUMAN_REVIEW |
| `INFORMATION_REQUIRED` (missing) | HUMAN_REVIEW |
| `LOGGING_REQUIRED` | EMIT (audit only) |

**Per-session evaluated checks (Phase 2 + 3):**

| ID | Severity | Description |
|----|----------|-------------|
| PDC-001 | HARD_BLOCK | Segment alignment verified (all including criteria met, no excluding) |
| PDC-002 | HARD_BLOCK | Data accuracy verification (stale data must be confirmed) |
| PDC-003 | HARD_BLOCK | No known unsuitability (vulnerability + hardship + arrears) |
| PDC-004 | HARD_BLOCK | Retail client classification confirmed |
| PDC-005 | HARD_BLOCK | Not a Pension Dashboard post-view service |
| PDC-006 | HARD_BLOCK | Directly authorised firm only (no appointed representatives) |
| PDC-007 | INFO_REQUIRED | Medium-materiality assumption consumer check completed (ML confidence ≥ 0.75) |
| DEL-001 | HARD_BLOCK | Mandatory 'targeted support' service label |
| DEL-002 | HARD_BLOCK | Not personalised advice — nature and limitations disclosure |
| DEL-003 | HARD_BLOCK | Segment characteristics disclosed to consumer |
| DEL-005 | DISCLOSURE | Capital at risk disclosure (investment products) |
| DEL-006 | HARD_BLOCK | MoneyHelper mandatory signpost (+ Pension Wise for pension domains) |
| DEL-012 | HARD_BLOCK | COBS 4 — fair, clear, not misleading |
| DEL-013 | LOGGING | Delivery audit record mandatory |
| DEL-014 | INFO_REQUIRED | No pension consolidation affirmative statement |

---

## FCA Compliance Invariants

| ID | Zone | Statement |
|----|------|-----------|
| INV-01 | Zone 1→2 | TraitGraph confirmed in Neo4j before Zone 2 starts |
| INV-02 | Zone 1.5 | ML prediction influences fill order only — never final segment |
| INV-03 | Zone 1.5 | Every SegmentHypothesis persisted (Neo4j or DLQ) before next turn |
| INV-04 | Zone 3 | No suggestion delivered unless all HARD_BLOCK checks pass |
| INV-05 | Zone 4 | Cloud Spanner write confirmed before consumer message released |
| INV-06 | Zone 4 | Consumer message from Jinja2 template only — no LLM-generated text |
| INV-07 | Zone 3 | `SUGGESTION_RULE_EVALUATED` signal emitted for every check |
| INV-08 | Ontology | No ontology node modified without GitOps PR + SMCR sign-off |
| INV-09 | Session | Consumer never re-asked about a trait already KNOWN in TraitGraph |
| INV-10 | Zone 4 | `ExplainabilityBundle.symbolic_trace` complete before Spanner write |

---

## EAMGP Signal Taxonomy

| Zone | Signal | Level |
|------|--------|-------|
| Zone 1 | `GRAPH_BUILD_START` / `GRAPH_BUILD_COMPLETE` | INFO |
| Zone 1 | `NEO4J_WRITE_HARD_FAIL` / `NEO4J_CIRCUIT_OPENED` | ERROR |
| Zone 1.5 | `SEG_PREDICT_START` / `SEG_PREDICT_COMPLETE` | INFO |
| Zone 1.5 | `SEG_PREDICT_DRIFT` / `SEG_GAP_REORDERED` | WARN |
| Zone 2 | `GAP_FILL_QUESTION_ASKED` / `GAP_FILL_ANSWERED` | INFO |
| Zone 2 | `SEGMENT_MATCHED` / `SEGMENT_NO_MATCH` / `SEGMENT_EXCLUDED` | INFO/WARN |
| Zone 3 | `SUGGESTION_CANDIDATES_RETRIEVED` | INFO |
| Zone 3 | `SUGGESTION_RULE_EVALUATED` (INV-07) | INFO |
| Zone 3 | `SUGGESTION_REJECTED` / `SUGGESTION_SUPPRESSED` | INFO/WARN |
| Zone 3 | `SUGGESTION_GATE_HUMAN_REVIEW` / `SUGGESTION_VALIDATED` | WARN/INFO |
| Zone 4 | `OUTPUT_CONSTRUCTED` / `AUDIT_WRITE_CONFIRMED` | INFO |
| Zone 4 | `CONSUMER_EXPLAIN_SERVED` | INFO |

---

## Known Limitations

| ID | Description | Fix |
|----|-------------|-----|
| ML-GAP-01 | `pension_contribution_pct`, `pension_pot_value` etc. are NaN for investment segments and investment features are NaN for pension segments. ML accuracy is ~39% on 14-class (>5× random baseline). Features are domain-specific by design. | Add a domain-routing layer before the segment predictor — one predictor per domain (4 classes each) would give ~70%+ per-domain accuracy. |
| ADK-01 | `zone2/agent.py` and `lead_agent.py` are 0% covered — require live Gemini LLM + GCP Vertex AI. | Needs GCP integration test environment. All 4 tool functions are 95% covered via FakeToolContext. |
| PROMPT-01 | Descriptive paraphrase variants ("I'm in my mid-thirties") coerce to `str` while numeric utterances coerce to `int`. Type consistency xfail expected. | Zone 0 DeBERTa entity extraction would normalise slot values before coercion. |
| VIS-01 | Visualiser has no authentication. | Add IAP (Identity-Aware Proxy) before exposing externally. |

---

## Production Deployment Notes

| Component | Production | Local equivalent |
|-----------|-----------|-----------------|
| Intent classifier (Zone 0) | DeBERTa-v3 on Vertex AI Prediction | Hardcoded intent from demo menu |
| TraitGraph persistence | Neo4j Causal Cluster (europe-west2) | Python dict in memory |
| ADK runner | GCP Agent Engine | FakeToolContext in tests |
| LLM | Gemini 2.0 Flash (Vertex AI) | Your keyboard |
| Audit log | Cloud Spanner | `audit_confirmed=False` (held) |
| Signal routing | Cloud Operations / BigQuery | In-memory SessionStore |

### Environment variables (production)

```bash
NEO4J_URI=bolt://neo4j-leader.europe-west2.internal:7687
NEO4J_USER=ts_agent
NEO4J_PASSWORD=<from Secret Manager>
GOOGLE_CLOUD_PROJECT=lbg-ts-platform-prod
SPANNER_INSTANCE=ts-audit
SPANNER_DATABASE=ts_events
PUBSUB_DLQ_TOPIC=projects/lbg-ts-platform-prod/topics/ts-dlq
GEMINI_MODEL=gemini-2.0-flash
```
