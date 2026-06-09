#!/usr/bin/env bash
set -e

cd ~/UCLA_Capstone
source .venv/bin/activate

MAIN_MODEL="meta-llama/Llama-3.2-3B-Instruct"

echo "Stopping old vLLM servers..."
pkill -f "vllm serve" || true
sleep 5

echo "Starting STANDARD Llama server 1 on port 8001..."
CUDA_VISIBLE_DEVICES=0 vllm serve "$MAIN_MODEL" \
  --host 127.0.0.1 \
  --port 8001 \
  --max-model-len 768 \
  --gpu-memory-utilization 0.40 \
  --generation-config vllm \
  --enforce-eager \
  --attention-backend TRITON_ATTN \
  --uvicorn-log-level info &

STD1_PID=$!
echo "Standard 1 PID: $STD1_PID"

sleep 30

echo ""
echo "Starting STANDARD Llama server 2 on port 8002..."
CUDA_VISIBLE_DEVICES=0 vllm serve "$MAIN_MODEL" \
  --host 127.0.0.1 \
  --port 8002 \
  --max-model-len 768 \
  --gpu-memory-utilization 0.40 \
  --generation-config vllm \
  --enforce-eager \
  --attention-backend TRITON_ATTN \
  --uvicorn-log-level info &

STD2_PID=$!
echo "Standard 2 PID: $STD2_PID"

echo ""
echo "Two standard servers are starting."
echo "Wait until you see application startup complete for both."
echo ""
echo "Health checks:"
echo "curl -i http://127.0.0.1:8001/health"
echo "curl -i http://127.0.0.1:8002/health"
echo ""
echo "Press Ctrl-C to stop watching, then kill servers manually if needed:"
echo "kill $STD1_PID $STD2_PID"

wait
