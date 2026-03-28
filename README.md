# ClinIQ — Knowledge Distillation for Offline Clinical Triage

> An offline-first AI clinical triage assistant built for community health workers and nurses in under-resourced clinics across sub-Saharan Africa and similar low-resource settings. ClinIQ compresses a large teacher model into a deployable 0.5B-parameter student that runs on commodity hardware — zero internet, zero cloud cost, zero compromise on clinical safety.

---

## The Problem

Community health workers in rural clinics face a critical gap: no specialist access, unreliable connectivity, and life-or-death triage decisions made with minimal support. ClinIQ bridges that gap by running a clinically-aligned AI assistant entirely on-device — no API calls, no subscriptions, no infrastructure dependency.

---

## Architecture Overview

```
╔══════════════════════════════════════════════════════════════════════╗
║                        TRAINING PIPELINE                            ║
║                    (Google Colab T4 / SageMaker Studio Lab)          ║
║                                                                      ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │  Stage C — Teacher SFT                                       │    ║
║  │  Model:    Qwen/Qwen3.5-9B (base, no adapter)               │    ║
║  │  Method:   Unsloth + LoRA (r=16, α=32)                      │    ║
║  │  Config:   3 epochs · LR=2e-4 · cosine · batch=2×8          │    ║
║  │  Output:   Teacher LoRA adapter → S3 checkpoints/stage_c/   │    ║
║  └──────────────────────────┬──────────────────────────────────┘    ║
║                             │ teacher soft labels                    ║
║  ┌──────────────────────────▼──────────────────────────────────┐    ║
║  │  Stage B — Student SFT                                       │    ║
║  │  Model:    Qwen/Qwen2.5-0.5B-Instruct                       │    ║
║  │  Method:   Unsloth + LoRA (r=8, α=16)                       │    ║
║  │  Config:   5 epochs · LR=3e-4 · cosine · batch=4×4          │    ║
║  │  Output:   Student SFT adapter → S3 checkpoints/stage_b/    │    ║
║  └──────────────────────────┬──────────────────────────────────┘    ║
║                             │ student foundation weights             ║
║  ┌──────────────────────────▼──────────────────────────────────┐    ║
║  │  Stage A — KL Logit Distillation (Raw PyTorch)               │    ║
║  │  Loss:  α·KL(student‖teacher)·T² + (1−α)·CE(student,labels) │    ║
║  │  T=2.0 · α=0.5 · warmup=50 steps · batch=1×16               │    ║
║  │  VRAM:  8-bit AdamW · grad ckpt · AMP · cache flush/10 steps │    ║
║  │  Output: Distilled adapter → S3 checkpoints/stage_a/        │    ║
║  └──────────────────────────┬──────────────────────────────────┘    ║
╚═════════════════════════════╪════════════════════════════════════════╝
                              │ S3 PutObject event → EventBridge
╔═════════════════════════════▼════════════════════════════════════════╗
║                   EVENT-DRIVEN ORCHESTRATION (AWS)                   ║
║                                                                      ║
║  S3 (cliniq-distillation)                                            ║
║    ├── checkpoints/stage_c/  ──► EventBridge ──► cliniq-notify λ    ║
║    ├── checkpoints/stage_b/  ──► EventBridge ──► cliniq-notify λ    ║
║    ├── checkpoints/stage_a/  ──► EventBridge ──► cliniq-notify λ    ║
║    └── metrics/stage_a/      ──► EventBridge ──► cliniq-compare λ   ║
║                                        │                             ║
║  cliniq-compare-models Lambda:         │                             ║
║    1. Fetch metrics (stage_a/b/c)      │                             ║
║    2. Compare eval_loss across stages  │                             ║
║    3. Fallback: Stage B if SFT < KD    │                             ║
║    4. Push winning LoRA → HuggingFace  │                             ║
║    5. Write comparison_report.json     ▼                             ║
║                                  ntfy.sh push                        ║
║                                  (phone/desktop)                     ║
╚══════════════════════════════════════════════════════════════════════╝
                              │ winning LoRA adapter (~100MB)
╔═════════════════════════════▼════════════════════════════════════════╗
║                         DEPLOYMENT                                   ║
║                                                                      ║
║  HuggingFace Hub (LoRA adapter only — fits Lambda /tmp)              ║
║       │                                                              ║
║       ▼                                                              ║
║  Ollama (Qwen2.5:0.5b base + LoRA · Q4_K_M quantized)               ║
║       │  temp=0.3 · top_p=0.85 · top_k=40 · num_predict=1024        ║
║       ▼                                                              ║
║  FastAPI serve.py  ──  POST /triage  ──  TriageResponse JSON         ║
║       │                                                              ║
║  Docker Compose (cliniq-ollama + cliniq-api · healthcheck chain)     ║
║       │                                                              ║
║  Clinic Endpoint: http://localhost:8000                              ║
║  Runs on ≤4GB RAM · Zero internet · $0/month                        ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## Knowledge Distillation — Technical Deep Dive

ClinIQ uses a **3-stage reverse-ordered distillation pipeline** (C → B → A). Each stage builds on the previous, progressively transferring clinical reasoning from a large teacher into a 0.5B student that fits on a clinic laptop.

### Stage C — Teacher Supervised Fine-Tuning

The teacher (`Qwen/Qwen3.5-9B`) is fine-tuned on curated clinical triage data using [Unsloth](https://github.com/unslothai/unsloth) for memory-efficient LoRA training. The base model is used directly — no external adapter is loaded, as GGUF-format repos carry no safetensors adapter weights.

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen3.5-9B` |
| Teacher adapter | `null` — base model used directly |
| LoRA rank / alpha | r=16, α=32 |
| Target modules | q/k/v/o/gate/up/down projections |
| Epochs | 3 |
| Learning rate | 2e-4 (cosine decay, warmup_ratio=0.1) |
| Effective batch | 2 per device × 8 grad accum = 16 |
| Precision | fp16 + gradient checkpointing |
| Checkpoint kept | 2 (best by `eval_loss`) |
| Output | `outputs/stage_c/final_adapter/` → S3 |

### Stage B — Student Supervised Fine-Tuning

The student (`Qwen2.5-0.5B-Instruct`) is independently fine-tuned on the same clinical dataset. This domain-grounded foundation prevents the student from starting KL distillation cold — it already understands clinical language before being asked to match the teacher's probability distributions.

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| LoRA rank / alpha | r=8, α=16 (smaller — student capacity) |
| Epochs | 5 |
| Learning rate | 3e-4 (cosine decay, warmup_ratio=0.1) |
| Effective batch | 4 per device × 4 grad accum = 16 |
| Output | `outputs/stage_b/final_adapter/` → S3 |

### Stage A — KL Logit Distillation

The core distillation stage, implemented in raw PyTorch (no Unsloth). The student learns to match the teacher's **full soft probability distribution** over the entire vocabulary — not just the argmax token — transferring the teacher's uncertainty, calibration, and "dark knowledge" (Hinton et al., 2015).

**Loss function:**

```
L = α · KL(softmax(z_s/T) ‖ softmax(z_t/T)) · T²  +  (1−α) · CE(z_s, y)
```

| Symbol | Meaning |
|---|---|
| `z_s`, `z_t` | Student and teacher logits `[batch, seq_len, vocab]` |
| `T = 2.0` | Temperature — softens distributions, amplifies inter-class signal |
| `α = 0.5` | Balance between soft-target KL and hard-label CE |
| `T²` scaling | Preserves gradient magnitude under temperature scaling (Hinton et al., 2015) |
| `ignore_index=-100` | Padding tokens excluded from CE loss |

**Vocab size mismatch:** Teacher and student may have different vocabulary sizes. Stage A clips both logit tensors to `min(vocab_student, vocab_teacher)` before computing KL divergence — no padding, no error.

**Training loop details:**
- Cosine LR scheduler with 50-step linear warmup
- `GradScaler` for AMP loss scaling
- Gradient norm clipping at 1.0
- Per-batch VRAM usage logged every 10 steps
- `torch.cuda.empty_cache()` every 10 steps

**VRAM Safety (T4 15GB budget):**

| Technique | Implementation |
|---|---|
| 8-bit AdamW | `bitsandbytes.optim.AdamW8bit` |
| Gradient checkpointing | `student_model.gradient_checkpointing_enable()` |
| Teacher in 4-bit | `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)` |
| Teacher frozen | `param.requires_grad = False` — no gradients stored |
| Mixed precision | `torch.amp.autocast("cuda")` + `GradScaler` |
| Cache flush | `torch.cuda.empty_cache()` every 10 steps |
| Batch size | 1 × 16 grad accum = effective 16 |

---

## Data Pipeline

```
data/
├── prepare.py              # Master pipeline: fetch + clean + split + save
├── fetch_external.py       # Pulls external clinical datasets (MedQA, etc.)
├── fetch_regulations.py    # Scrapes regulatory documents
├── dataset.py              # HuggingFace Dataset loaders for SFT + distillation
├── regulations.py          # Regulatory text processing
├── prompts/
│   ├── system_prompt.txt   # WHO IMAI/IMCI-aligned clinical system prompt
│   └── templates.py        # Chat template formatters + vignette templates
└── processed/
    ├── train.jsonl          # Training split (messages format)
    ├── eval.jsonl           # Evaluation split
    ├── external_data.jsonl  # External clinical QA data
    ├── regulations.jsonl    # Regulatory compliance data
    └── regulations_*.txt    # HIPAA, POPIA, FHIR R4, SA NHA, SA STG/EML
```

**Regulatory compliance data:**

| Document | Scope |
|---|---|
| HIPAA | US health data privacy |
| POPIA | South Africa Protection of Personal Information Act |
| FHIR R4 | HL7 interoperability standard |
| SA National Health Act | South African public health law |
| SA STG/EML | Standard Treatment Guidelines & Essential Medicines List |

**Prompt format** (OpenAI-compatible chat template):
```json
{
  "messages": [
    {"role": "system",    "content": "<WHO IMAI/IMCI clinical system prompt>"},
    {"role": "user",      "content": "<patient symptom description>"},
    {"role": "assistant", "content": "<structured triage response>"}
  ]
}
```

**Vignette templates** (`data/prompts/templates.py`):

Three structured templates drive synthetic training data generation:

- `TRIAGE_VIGNETTE_TEMPLATE` — standard clinic presentation (age, sex, chief complaint, vitals, history)
- `FOLLOW_UP_TEMPLATE` — returning patient with original diagnosis and current treatment
- `EMERGENCY_TEMPLATE` — community health worker field report with resource and distance context

---

## AWS Event-Driven Orchestration

All pipeline events are fully automated via AWS serverless infrastructure — no manual monitoring required.

### S3 Bucket: `cliniq-distillation`

```
cliniq-distillation/
├── checkpoints/
│   ├── stage_c/stage_c_checkpoint.tar.gz   # Teacher LoRA adapter
│   ├── stage_b/stage_b_checkpoint.tar.gz   # Student SFT adapter
│   └── stage_a/stage_a_checkpoint.tar.gz   # Distilled student adapter
└── metrics/
    ├── stage_c/eval_metrics.json
    ├── stage_b/eval_metrics.json
    ├── stage_a/eval_metrics.json
    └── comparison_report.json              # Winner selection report
```

Lifecycle policies: checkpoints expire after **30 days**, metrics after **90 days**. Incomplete multipart uploads aborted after **1 day**.

### EventBridge Rules

| Rule | S3 Trigger | Targets |
|---|---|---|
| `cliniq-checkpoint-uploaded` | `ObjectCreated` on `checkpoints/*` | `cliniq-notify` λ |
| `cliniq-metrics-uploaded` | `ObjectCreated` on `metrics/stage_a/*` | `cliniq-compare-models` λ + `cliniq-notify` λ |

### Lambda: `cliniq-compare-models`

Triggered automatically when Stage A metrics land in S3. Performs:

1. Fetches `eval_metrics.json` for all three stages from S3
2. Compares `eval_loss` — distilled student (A) vs SFT student (B) vs teacher (C)
3. **Fallback logic:** Stage A is preferred; if `stage_b.eval_loss < stage_a.eval_loss`, Stage B wins
4. Downloads the winning LoRA adapter archive from S3 (`~100MB tar.gz`)
5. Extracts to Lambda `/tmp`, pushes to HuggingFace Hub via `HfApi.upload_folder()`
6. Writes `metrics/comparison_report.json` back to S3
7. Sends formatted comparison summary via ntfy.sh

> Only the LoRA adapter (~100MB) is pushed to HuggingFace — not the full merged model — to stay within Lambda's `/tmp` storage and 15-minute execution limits.

### Lambda: `cliniq-notify`

Lightweight notification relay. Parses the S3 object key to determine pipeline stage and sends formatted push notifications to ntfy.sh:

| Key pattern | Title | Priority |
|---|---|---|
| `checkpoints/stage_c/*` | 📦 Stage C: Teacher SFT | `high` |
| `checkpoints/stage_b/*` | 📦 Stage B: Student SFT | `high` |
| `checkpoints/stage_a/*` | 📦 Stage A: KL Distillation | `high` |
| `metrics/*` | stage metrics saved | `default` |
| `comparison_report*` | 🏆 Model Comparison | `default` |

### Notifications (ntfy.sh)

Real-time push notifications to phone/desktop for every pipeline event — no polling, no dashboard. Configure in `.env`:

```
NTFY_TOPIC=cliniq-pipeline
NTFY_SERVER=https://ntfy.sh
```

---

## Evaluation Metrics

ClinIQ tracks four evaluation dimensions, implemented in `utils/metrics.py`:

| Metric | Function | Description |
|---|---|---|
| **Perplexity** | `compute_perplexity()` | `exp(avg_cross_entropy_loss)` — lower is better |
| **ROUGE-L F1** | `compute_rouge_l()` | LCS-based content overlap between student and teacher outputs |
| **Token Accuracy** | `compute_token_accuracy()` | Top-1 token agreement between student and teacher logits |
| **Clinical Safety Score** | `compute_clinical_safety_score()` | Composite of `safety_score`, `referral_rate`, `disclaimer_rate` |

**Clinical safety sub-metrics:**

| Sub-metric | What it measures |
|---|---|
| `safety_score` | Fraction of outputs that recognise emergency/red-flag keywords |
| `referral_rate` | Fraction of outputs that include a referral recommendation |
| `disclaimer_rate` | Fraction of outputs that carry an appropriate clinical disclaimer |

A model that fails to recommend referral for a life-threatening presentation is penalised regardless of its perplexity — clinical safety is a hard constraint, not a soft optimisation target.

---

## Deployment

### Option 1 — Docker Compose (recommended for clinics)

```bash
docker compose -f deploy/docker-compose.yml up -d
```

Starts two services with a healthcheck dependency chain:
- `cliniq-ollama` — Ollama runtime, persistent model storage via named volume `ollama-data`
- `cliniq-api` — FastAPI server, starts only after Ollama passes `service_healthy`

### Option 2 — Manual

```bash
ollama create cliniq -f deploy/Modelfile
python deploy/serve.py
```

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Ollama connectivity + model status |
| `POST` | `/triage` | Patient triage assessment |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/` | API root + endpoint index |

**Triage request:**
```json
{
  "patient_description": "34-year-old female, persistent cough 3 weeks, blood-tinged sputum, night sweats, weight loss. HIV-positive on ART. Household TB contact.",
  "urgency_context": "clinic"
}
```

**Triage response:**
```json
{
  "triage_level": "URGENT",
  "assessment": "...",
  "model": "cliniq",
  "timestamp": "2026-03-28T08:33:11",
  "disclaimer": "⚠️ ClinIQ is a decision-support tool only. All assessments require validation by a qualified healthcare professional.",
  "processing_time_ms": 1842.5
}
```

**Triage level parsing:** The API parses the model's free-text output for `EMERGENCY`/`RED` → `URGENT`/`YELLOW` → defaults to `ROUTINE`. This is intentionally conservative — ambiguous outputs default to the lower-urgency classification.

### Inference Parameters (Modelfile)

| Parameter | Value | Rationale |
|---|---|---|
| `temperature` | 0.3 | Low — clinical precision over creativity |
| `top_p` | 0.85 | Nucleus sampling for coherent outputs |
| `top_k` | 40 | Vocabulary restriction |
| `repeat_penalty` | 1.1 | Prevents repetitive safety disclaimers |
| `num_predict` | 1024 | Sufficient for structured 6-section response |

---

## Clinical Scope

**Triage levels:** EMERGENCY (Red) · URGENT (Yellow) · ROUTINE (Green)

**Clinical domains:**
- Tuberculosis (TB) — cough >2 weeks, night sweats, weight loss, contact history
- HIV/AIDS — opportunistic infections, ART adherence, PMTCT protocols
- Malaria — fever patterns, RDT interpretation, antimalarial protocols
- Maternal Health — antenatal danger signs, delivery complications, postnatal care
- Paediatric/IMCI — childhood illness assessment, growth monitoring, immunization

**Guidelines alignment:** WHO IMAI (Integrated Management of Adult Illness) · WHO IMCI (Integrated Management of Childhood Illness)

**Structured response format** (every output):
1. Triage level with justification
2. Key findings
3. Differential diagnoses (top 2–3)
4. Recommended actions
5. Referral decision
6. Safety flags

**Hard safety rules baked into the system prompt:**
- Always flag emergency symptoms regardless of presenting complaint
- Never recommend stopping ART or TB treatment
- Always recommend HIV testing in high-prevalence settings with suggestive symptoms
- Children under 5: always assess IMCI danger signs
- Pregnant women: always screen for pre-eclampsia danger signs
- When uncertain: always refer to a higher-level facility

---

## Project Structure

```
Knowldge-Distillation103/
├── data/
│   ├── prepare.py                  # Master data pipeline
│   ├── dataset.py                  # HuggingFace Dataset loaders (SFT + distillation)
│   ├── fetch_external.py           # External clinical dataset fetcher (MedQA etc.)
│   ├── fetch_regulations.py        # Regulatory document scraper
│   ├── regulations.py              # Regulatory text processor
│   ├── prompts/
│   │   ├── system_prompt.txt       # WHO IMAI/IMCI clinical system prompt
│   │   └── templates.py            # Chat formatters + vignette templates
│   └── processed/                  # Generated training data (gitignored)
│
├── stage_c/
│   ├── train.py                    # Teacher SFT (Unsloth + LoRA)
│   ├── train_colab.ipynb           # Colab notebook variant
│   └── config.yaml                 # Teacher training config
│
├── stage_b/
│   ├── train.py                    # Student SFT (Unsloth + LoRA)
│   └── config.yaml                 # Student SFT config
│
├── stage_a/
│   ├── distill.py                  # KL logit distillation (raw PyTorch)
│   └── config.yaml                 # Distillation config (T, α, VRAM safety)
│
├── utils/
│   ├── config.py                   # YAML config loader
│   ├── checkpoint.py               # S3 upload/download + cleanup
│   ├── metrics.py                  # Perplexity, ROUGE-L, token accuracy, clinical safety
│   └── notify.py                   # ntfy.sh notification helpers
│
├── infra/
│   ├── setup_aws.sh                # One-shot AWS infrastructure provisioning
│   ├── s3/bucket_policy.json       # S3 bucket config + lifecycle rules
│   ├── eventbridge/rules.json      # EventBridge rule definitions
│   └── lambda/
│       ├── compare_models/handler.py   # Model comparison + HF push Lambda
│       └── notify/handler.py           # ntfy.sh notification Lambda
│
├── deploy/
│   ├── Modelfile                   # Ollama model definition
│   ├── serve.py                    # FastAPI inference server
│   ├── Dockerfile                  # Container image for cliniq-api
│   └── docker-compose.yml          # Ollama + FastAPI stack (healthcheck chain)
│
├── tests/
│   ├── test_dataset.py             # Data pipeline tests
│   ├── test_checkpoint.py          # S3 checkpoint tests (moto)
│   └── test_serve.py               # FastAPI endpoint tests
│
├── .github/
│   ├── workflows/sync-to-hf.yml    # GitHub Actions → HuggingFace Hub sync
│   └── dependabot.yml              # Dependency update automation
│
├── requirements.txt
├── Makefile
└── .env.example
```

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd Knowldge-Distillation103
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — set AWS credentials, HF_TOKEN, NTFY_TOPIC, model names

# 3. Provision AWS infrastructure (one-time)
bash infra/setup_aws.sh

# 4. Prepare training data
python data/prepare.py

# 5. Run training pipeline (Colab T4 / SageMaker Studio Lab)
python stage_c/train.py          # Teacher SFT  (~2–3 hrs on T4)
python stage_b/train.py          # Student SFT  (~1–2 hrs on T4)
python stage_a/distill.py        # KL Distill   (~1–2 hrs on T4)

# Validate configs without GPU first:
python stage_c/train.py --dry-run
python stage_b/train.py --dry-run
python stage_a/distill.py --dry-run   # also asserts VRAM safety flags are set

# 6. Deploy locally
docker compose -f deploy/docker-compose.yml up -d
# or manually:
ollama create cliniq -f deploy/Modelfile
python deploy/serve.py

# 7. Test the API
curl -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"patient_description": "5-year-old with high fever 40C, convulsions, unable to drink. Rural clinic, 2 hours from hospital."}'
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | AWS credentials | — |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials | — |
| `AWS_DEFAULT_REGION` | AWS region | `us-east-1` |
| `S3_BUCKET_NAME` | S3 bucket for checkpoints/metrics | `cliniq-distillation` |
| `HF_TOKEN` | HuggingFace token for model push | — |
| `HF_REPO_ID` | HuggingFace repo for the winning adapter | — |
| `NTFY_TOPIC` | ntfy.sh topic for push notifications | `cliniq-pipeline` |
| `NTFY_SERVER` | ntfy.sh server URL | `https://ntfy.sh` |
| `TEACHER_MODEL` | Teacher base model | `Qwen/Qwen3.5-9B` |
| `TEACHER_ADAPTER` | Teacher LoRA adapter path (optional) | `null` |
| `STUDENT_MODEL` | Student base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| `OLLAMA_HOST` | Ollama API host | `http://localhost:11434` |
| `SERVE_PORT` | FastAPI server port | `8000` |

---

## Hardware Requirements

| Stage | Minimum | Recommended |
|---|---|---|
| Stage C (Teacher SFT) | 15GB VRAM (T4) | A100 40GB |
| Stage B (Student SFT) | 8GB VRAM | T4 15GB |
| Stage A (KL Distill) | 15GB VRAM (T4, all VRAM safety flags on) | A100 40GB |
| Inference (Ollama) | 4GB RAM, no GPU | Any clinic laptop |

All training stages are designed to run on a **free Google Colab T4 GPU** with the VRAM safety configurations enabled.

---

## CI/CD

- `.github/workflows/sync-to-hf.yml` — Automatically syncs repository to HuggingFace Hub on every push
- `.github/dependabot.yml` — Automated dependency updates
- Lambda `cliniq-compare-models` — Automated model evaluation and adapter promotion on every completed distillation run — no human intervention required

---

## License

MIT

---

> ⚠️ **Clinical Disclaimer**: ClinIQ is a decision-support tool only. All outputs require validation by a qualified healthcare professional. ClinIQ does not provide definitive diagnoses and is not a substitute for clinical judgment. Always refer patients with serious or uncertain presentations to a higher-level facility.
