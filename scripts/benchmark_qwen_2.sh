cd "$(dirname "$0")/.."
source .venv/bin/activate

# Clear proxy variables that can force httpx into a SOCKS transport without the required dependency.
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy FTP_PROXY ftp_proxy GIT_SSH_COMMAND GRPC_PROXY grpc_proxy RSYNC_PROXY

CUDA_VISIBLE_DEVICES=3,7 python main.py --config configs/qwen_2.yaml --benchmark gsm8k --max-prompts 20
CUDA_VISIBLE_DEVICES=3,7 python main.py --config configs/qwen_2.yaml --benchmark humaneval --max-prompts 20
CUDA_VISIBLE_DEVICES=3,7 python main.py --config configs/qwen_2.yaml --benchmark mt-bench --max-prompts 20