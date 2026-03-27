# ClinIQ — Knowledge Distillation for Offline Clinical Triage

An offline-first AI clinical triage assistant built for community health workers and nurses in under-resourced clinics. Uses a 3-stage knowledge distillation pipeline to compress a 1.5B-parameter teacher model into a 0.5B student that runs on commodity hardware via [Ollama](https://ollama.ai).

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                         │
│                                                              │
│  Stage C (Teacher SFT)    Qwen2.5-1.5B + Unsloth + LoRA     │
│         ↓                                                    │
│  Stage B (Student SFT)    Qwen2.5-0.5B + Unsloth + LoRA     │
│         ↓                                                    │
│  Stage A (KL Distill)     Raw PyTorch KL Divergence          │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│              EVENT-DRIVEN ORCHESTRATION                      │
│                                                              │
│  S3 PutObject → EventBridge → Lambda (compare + notify)      │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                  DEPLOYMENT                                  │
│                                                              │
│  Ollama (Q4 quantized) → FastAPI → Clinic Endpoint           │
│  Runs on ≤4GB RAM, zero internet, $0/month                   │
└──────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd Knowldge-Distillation103
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your AWS credentials, HF token, ntfy topic

# 3. Prepare data
python data/prepare.py

# 4. Run training stages (on SageMaker Studio Lab)
python stage_c/train.py    # Teacher SFT
python stage_b/train.py    # Student SFT
python stage_a/distill.py  # KL Distillation

# 5. Deploy locally
ollama create cliniq -f deploy/Modelfile
python deploy/serve.py
```

## Project Structure

```
├── data/           # Data pipeline & prompt templates
├── utils/          # Shared utilities (checkpoint, metrics, notify)
├── stage_c/        # Teacher SFT with Unsloth
├── stage_b/        # Student SFT with Unsloth
├── stage_a/        # KL Logit Distillation (raw PyTorch)
├── infra/          # AWS Lambda, EventBridge, S3 configs
├── deploy/         # Ollama + FastAPI deployment
└── tests/          # Unit tests
```

## Clinical Scope

- **Triage**: Emergency / Urgent / Routine classification
- **Domains**: TB, HIV, malaria, maternal health, paediatric illness
- **Guidelines**: WHO IMAI/IMCI aligned
- **Output**: Structured JSON with differentials, actions, and disclaimers

> ⚠️ **Disclaimer**: ClinIQ is a decision-support tool. All outputs require clinical validation by a qualified healthcare professional.

## License

MIT
