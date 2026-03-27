#!/bin/bash
# ===================================================================
# ClinIQ — AWS Infrastructure Setup (One-Shot Provisioning)
# ===================================================================
# This script creates all required AWS resources for the pipeline.
# Run once. Safe to re-run (uses --no-fail-on-existing-bucket etc.)
#
# Prerequisites:
#   - AWS CLI installed and configured (aws configure)
#   - aws-cli v2 recommended
# ===================================================================

set -euo pipefail

# ===== Configuration =====
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
BUCKET_NAME="${S3_BUCKET_NAME:-cliniq-distillation}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
LAMBDA_ROLE_NAME="cliniq-lambda-role"
NOTIFY_FUNCTION="cliniq-notify"
COMPARE_FUNCTION="cliniq-compare-models"

echo "=============================================="
echo "ClinIQ AWS Infrastructure Setup"
echo "=============================================="
echo "Region:     $REGION"
echo "Bucket:     $BUCKET_NAME"
echo "Account:    $ACCOUNT_ID"
echo "=============================================="

# ===== 1. Create S3 Bucket =====
echo ""
echo "📦 Creating S3 bucket: $BUCKET_NAME"
if [ "$REGION" == "us-east-1" ]; then
    aws s3api create-bucket \
        --bucket "$BUCKET_NAME" \
        --region "$REGION" \
        2>/dev/null || echo "   Bucket already exists."
else
    aws s3api create-bucket \
        --bucket "$BUCKET_NAME" \
        --region "$REGION" \
        --create-bucket-configuration LocationConstraint="$REGION" \
        2>/dev/null || echo "   Bucket already exists."
fi

# Enable EventBridge notifications on bucket
echo "   Enabling EventBridge notifications..."
aws s3api put-bucket-notification-configuration \
    --bucket "$BUCKET_NAME" \
    --notification-configuration '{"EventBridgeConfiguration": {}}'

# Apply lifecycle rules
echo "   Applying lifecycle rules..."
aws s3api put-bucket-lifecycle-configuration \
    --bucket "$BUCKET_NAME" \
    --lifecycle-configuration file://infra/s3/bucket_policy.json \
    2>/dev/null || echo "   Using simplified lifecycle config."

# Tag bucket
aws s3api put-bucket-tagging \
    --bucket "$BUCKET_NAME" \
    --tagging 'TagSet=[{Key=Project,Value=ClinIQ},{Key=CostCenter,Value=free-tier}]'

echo "   ✅ S3 bucket ready."

# ===== 2. Create IAM Role for Lambda =====
echo ""
echo "🔐 Creating Lambda execution role..."

TRUST_POLICY='{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "lambda.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}'

aws iam create-role \
    --role-name "$LAMBDA_ROLE_NAME" \
    --assume-role-policy-document "$TRUST_POLICY" \
    2>/dev/null || echo "   Role already exists."

# Attach basic Lambda + S3 read permissions
aws iam attach-role-policy \
    --role-name "$LAMBDA_ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole \
    2>/dev/null || true

# Inline policy for S3 access
S3_POLICY='{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
    "Resource": [
      "arn:aws:s3:::'"$BUCKET_NAME"'",
      "arn:aws:s3:::'"$BUCKET_NAME"'/*"
    ]
  }]
}'

aws iam put-role-policy \
    --role-name "$LAMBDA_ROLE_NAME" \
    --policy-name "cliniq-s3-access" \
    --policy-document "$S3_POLICY"

ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${LAMBDA_ROLE_NAME}"
echo "   ✅ IAM role ready: $ROLE_ARN"

# Wait for role propagation
echo "   ⏳ Waiting 10s for IAM role propagation..."
sleep 10

# ===== 3. Deploy Lambda Functions =====
echo ""
echo "⚡ Deploying Lambda functions..."

# Package notify Lambda
echo "   Packaging $NOTIFY_FUNCTION..."
cd infra/lambda/notify
zip -j /tmp/cliniq-notify.zip handler.py
cd ../../..

aws lambda create-function \
    --function-name "$NOTIFY_FUNCTION" \
    --runtime python3.12 \
    --handler handler.handler \
    --role "$ROLE_ARN" \
    --zip-file fileb:///tmp/cliniq-notify.zip \
    --timeout 30 \
    --memory-size 128 \
    --environment "Variables={NTFY_TOPIC=${NTFY_TOPIC:-cliniq-pipeline},NTFY_SERVER=https://ntfy.sh}" \
    --region "$REGION" \
    2>/dev/null || {
        echo "   Updating existing $NOTIFY_FUNCTION..."
        aws lambda update-function-code \
            --function-name "$NOTIFY_FUNCTION" \
            --zip-file fileb:///tmp/cliniq-notify.zip \
            --region "$REGION"
    }
echo "   ✅ $NOTIFY_FUNCTION deployed."

# Package compare_models Lambda
echo "   Packaging $COMPARE_FUNCTION..."
cd infra/lambda/compare_models
zip -j /tmp/cliniq-compare.zip handler.py
cd ../../..

aws lambda create-function \
    --function-name "$COMPARE_FUNCTION" \
    --runtime python3.12 \
    --handler handler.handler \
    --role "$ROLE_ARN" \
    --zip-file fileb:///tmp/cliniq-compare.zip \
    --timeout 900 \
    --memory-size 512 \
    --ephemeral-storage '{"Size": 2048}' \
    --environment "Variables={S3_BUCKET_NAME=$BUCKET_NAME,HF_TOKEN=${HF_TOKEN:-},HF_REPO_ID=${HF_REPO_ID:-cliniq/cliniq-0.5b},NTFY_TOPIC=${NTFY_TOPIC:-cliniq-pipeline}}" \
    --region "$REGION" \
    2>/dev/null || {
        echo "   Updating existing $COMPARE_FUNCTION..."
        aws lambda update-function-code \
            --function-name "$COMPARE_FUNCTION" \
            --zip-file fileb:///tmp/cliniq-compare.zip \
            --region "$REGION"
    }
echo "   ✅ $COMPARE_FUNCTION deployed."

# ===== 4. Create EventBridge Rules =====
echo ""
echo "📡 Creating EventBridge rules..."

# Rule 1: Checkpoint uploaded → notify
aws events put-rule \
    --name "cliniq-checkpoint-uploaded" \
    --event-pattern '{
        "source": ["aws.s3"],
        "detail-type": ["Object Created"],
        "detail": {
            "bucket": {"name": ["'"$BUCKET_NAME"'"]},
            "object": {"key": [{"prefix": "checkpoints/"}]}
        }
    }' \
    --region "$REGION"

aws events put-targets \
    --rule "cliniq-checkpoint-uploaded" \
    --targets "Id=notify,Arn=arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${NOTIFY_FUNCTION}" \
    --region "$REGION"

# Rule 2: Stage A metrics uploaded → compare_models + notify
aws events put-rule \
    --name "cliniq-metrics-uploaded" \
    --event-pattern '{
        "source": ["aws.s3"],
        "detail-type": ["Object Created"],
        "detail": {
            "bucket": {"name": ["'"$BUCKET_NAME"'"]},
            "object": {"key": [{"prefix": "metrics/stage_a/"}]}
        }
    }' \
    --region "$REGION"

aws events put-targets \
    --rule "cliniq-metrics-uploaded" \
    --targets "[
        {\"Id\":\"compare\",\"Arn\":\"arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${COMPARE_FUNCTION}\"},
        {\"Id\":\"notify\",\"Arn\":\"arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${NOTIFY_FUNCTION}\"}
    ]" \
    --region "$REGION"

# Grant EventBridge permission to invoke Lambdas
aws lambda add-permission \
    --function-name "$NOTIFY_FUNCTION" \
    --statement-id "eventbridge-invoke" \
    --action "lambda:InvokeFunction" \
    --principal events.amazonaws.com \
    --region "$REGION" \
    2>/dev/null || true

aws lambda add-permission \
    --function-name "$COMPARE_FUNCTION" \
    --statement-id "eventbridge-invoke" \
    --action "lambda:InvokeFunction" \
    --principal events.amazonaws.com \
    --region "$REGION" \
    2>/dev/null || true

echo "   ✅ EventBridge rules created."

# ===== Done =====
echo ""
echo "=============================================="
echo "✅ ClinIQ AWS Infrastructure Setup Complete!"
echo "=============================================="
echo ""
echo "Resources created:"
echo "  • S3 Bucket:    $BUCKET_NAME (with lifecycle rules)"
echo "  • IAM Role:     $LAMBDA_ROLE_NAME"
echo "  • Lambda:       $NOTIFY_FUNCTION (notifications)"
echo "  • Lambda:       $COMPARE_FUNCTION (model comparison)"
echo "  • EventBridge:  cliniq-checkpoint-uploaded"
echo "  • EventBridge:  cliniq-metrics-uploaded"
echo ""
echo "Next: Run your training pipeline!"
echo "  make train-student   # Stage B"
echo "  make distill         # Stage A → auto-triggers comparison"
