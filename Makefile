.PHONY: install data train-teacher train-student distill deploy test clean

# ===== Setup =====
install:
	pip install -r requirements.txt

# ===== Data Pipeline =====
data:
	python data/prepare.py

# ===== Training Stages =====
train-student:
	python stage_b/train.py

distill:
	python stage_a/distill.py

train-all: train-student distill

# ===== Deployment =====
deploy-model:
	ollama create cliniq -f deploy/Modelfile

serve:
	python deploy/serve.py

deploy: deploy-model serve

# ===== Testing =====
test:
	python -m pytest tests/ -v

test-data:
	python -m pytest tests/test_dataset.py -v

test-checkpoint:
	python -m pytest tests/test_checkpoint.py -v

test-serve:
	python -m pytest tests/test_serve.py -v

# ===== AWS Infrastructure =====
aws-setup:
	chmod +x infra/setup_aws.sh && bash infra/setup_aws.sh

# ===== Cleanup =====
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
	rm -rf data/processed/ data/raw/
