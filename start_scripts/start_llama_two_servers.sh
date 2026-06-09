#!/usr/bin/env bash
set -e

cd ~/UCLA_Capstone
source .venv/bin/activate

MAIN_MODEL="meta-llama/Llama-3.2-3B-Instruct"
DRAFT_MODEL="meta-llama/Llama-3.2-1B-Instruct"

echo "Stopping old vLLM servers..."
pkill -f "vllm serve" || true
sleep 5

echo "Starting REGULAR Llama server on port 8001..."
CUDA_VISIBLE_DEVICES=0 vllm serve "$MAIN_MODEL" \
  --host 127.0.0.1 \
  --port 8001 \
  --max-model-len 768 \
  --gpu-memory-utilization 0.43 \
  --generation-config vllm \
  --enforce-eager \
  --attention-backend TRITON_ATTN \
  --uvicorn-log-level error&

REG_PID=$!
echo "Regular PID: $REG_PID"

sleep 25

echo "Starting SPECULATIVE Llama server on port 8002..."
CUDA_VISIBLE_DEVICES=0 vllm serve "$MAIN_MODEL" \
  --host 127.0.0.1 \
  --port 8002 \
  --max-model-len 768 \
  --gpu-memory-utilization 0.5 \
  --generation-config vllm \
  --enforce-eager \
  --attention-backend TRITON_ATTN \
  --speculative-config "{\"method\":\"draft_model\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":5}" \
  --uvicorn-log-level error &

SPEC_PID=$!
echo "Speculative PID: $SPEC_PID"

echo ""
echo "Servers starting."
echo "Regular log:      regular_8001.log"
echo "Speculative log:  speculative_8002.log"
echo ""
echo "Check health:"
echo "curl -i http://127.0.0.1:8001/health"
echo "curl -i http://127.0.0.1:8002/health"

wait
