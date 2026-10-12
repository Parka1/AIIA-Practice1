#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VM_SSH_TARGET:-}" ]]; then
  printf 'Set VM_SSH_TARGET to user@public-ip or an SSH config alias.\n' >&2
  exit 2
fi

model_file="${VM_MODEL_FILE:-local_vm/models/yolov8n_960.onnx}"
if [[ ! -s "$model_file" ]]; then
  printf 'Model not found: %s\nExport it with python -m local_vm.export_vm_model first.\n' "$model_file" >&2
  exit 2
fi

ssh_args=(-o BatchMode=yes -o ServerAliveInterval=30)
scp_args=(-o BatchMode=yes)
if [[ -n "${VM_SSH_KEY:-}" ]]; then
  ssh_args+=(-i "$VM_SSH_KEY" -o IdentitiesOnly=yes)
  scp_args+=(-i "$VM_SSH_KEY" -o IdentitiesOnly=yes)
fi

config_dir="${XDG_CONFIG_HOME:-$HOME/.config}/aiia-practice1"
token_file="$config_dir/vm-upload-token"
install -d -m 700 "$config_dir"
if [[ ! -s "$token_file" ]]; then
  (umask 077; python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > "$token_file")
fi
chmod 600 "$token_file"

archive="$(mktemp "${TMPDIR:-/tmp}/aiia-part2.XXXXXX.tar.gz")"
trap 'rm -f "$archive"' EXIT
tar -czf "$archive" \
  local_vm/vm_inference.py \
  local_vm/vm_server.py \
  local_vm/aiia-part2.service \
  local_vm/aiia-part2-apache.conf \
  local_vm/requirements-vm.txt \
  local_vm/templates/vm_dashboard.html \
  "$model_file" \
  common/__init__.py \
  common/results.py

scp "${scp_args[@]}" "$archive" "$VM_SSH_TARGET:/tmp/aiia-part2-release.tar.gz"
ssh "${ssh_args[@]}" "$VM_SSH_TARGET" 'sudo bash -s' <<'REMOTE'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y apache2 python3-pip python3-venv

if ! id aiia-vm >/dev/null 2>&1; then
  sudo useradd --system --home-dir /var/lib/aiia-part2 --no-create-home \
    --shell /usr/sbin/nologin aiia-vm
fi
sudo install -d -o root -g aiia-vm -m 0750 /opt/aiia-part2
sudo tar -xzf /tmp/aiia-part2-release.tar.gz -C /opt/aiia-part2
sudo rm -f /tmp/aiia-part2-release.tar.gz
sudo chown -R root:aiia-vm /opt/aiia-part2
sudo find /opt/aiia-part2 -path /opt/aiia-part2/venv -prune -o \
  -type d -exec chmod 0750 {} +
sudo find /opt/aiia-part2 -path /opt/aiia-part2/venv -prune -o \
  -type f -exec chmod 0640 {} +
sudo chmod 0750 /opt/aiia-part2/local_vm/models
sudo install -d -o aiia-vm -g aiia-vm -m 0750 /var/lib/aiia-part2

if [[ ! -x /opt/aiia-part2/venv/bin/python ]]; then
  sudo python3 -m venv --system-site-packages /opt/aiia-part2/venv
fi
sudo /opt/aiia-part2/venv/bin/python -m pip install --disable-pip-version-check \
  --no-cache-dir -r /opt/aiia-part2/local_vm/requirements-vm.txt
sudo -u aiia-vm /opt/aiia-part2/venv/bin/python -c \
  'import cv2, numpy, flask, psutil; print("OpenCV", cv2.__version__, "NumPy", numpy.__version__)'

sudo install -o root -g aiia-vm -m 0640 \
  /opt/aiia-part2/local_vm/aiia-part2.service /etc/systemd/system/aiia-part2.service
sudo install -o root -g root -m 0644 \
  /opt/aiia-part2/local_vm/aiia-part2-apache.conf /etc/apache2/sites-available/aiia-part2.conf
if [[ -f /var/www/html/index.html && ! -f /var/www/html/index.html.before-aiia-part2 ]]; then
  sudo cp -a /var/www/html/index.html /var/www/html/index.html.before-aiia-part2
fi
sudo a2enmod proxy proxy_http
sudo a2dissite 000-default.conf >/dev/null 2>&1 || true
sudo a2ensite aiia-part2.conf
sudo apache2ctl configtest

sudo systemctl daemon-reload
sudo systemctl enable aiia-part2.service
REMOTE

{
  printf 'AIIA_VM_UPLOAD_TOKEN=%s\n' "$(cat "$token_file")"
  printf 'AIIA_MODEL_PATH=/opt/aiia-part2/local_vm/models/yolov8n_960.onnx\n'
  printf 'AIIA_DATA_DIR=/var/lib/aiia-part2\n'
  printf 'AIIA_VM_HOURLY_RATE_USD=%s\n' "${AIIA_VM_HOURLY_RATE_USD:-0.0114}"
} | ssh "${ssh_args[@]}" "$VM_SSH_TARGET" \
  'sudo install -o root -g aiia-vm -m 0640 /dev/stdin /etc/aiia-part2.env && sudo systemctl restart aiia-part2.service && sudo systemctl reload apache2'

printf 'Deployment complete. Token stored at %s\n' "$token_file"
