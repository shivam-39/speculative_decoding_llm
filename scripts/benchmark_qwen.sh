cd "$(dirname "$0")/.."
source .venv/bin/activate

# Clear proxy variables that can force httpx into a SOCKS transport without the required dependency.
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy FTP_PROXY ftp_proxy GIT_SSH_COMMAND GRPC_PROXY grpc_proxy RSYNC_PROXY

python main.py --config configs/qwen.yaml --benchmark gsm8k
python main.py --config configs/qwen.yaml --benchmark humaneval
python main.py --config configs/qwen.yaml --benchmark mt-bench