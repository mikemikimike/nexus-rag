#!/usr/bin/env bash
# Run as root on the VM, from the copied project (default /opt/local-rag).
# Needs: vendor/qdrant (binary), wheelhouse/ + requirements.lock (or an internal PyPI), python3-venv.
set -euo pipefail
APP="${APP:-/opt/local-rag}"
[ "$(id -u)" = 0 ] || { echo "run as root"; exit 1; }
[ -d "$APP" ] || { echo "$APP not found (copy the project there first)"; exit 1; }
[ -f "$APP/vendor/qdrant" ] || { echo "missing $APP/vendor/qdrant (run scripts/download_qdrant.sh on a connected machine)"; exit 1; }
rnd() { python3 -c "import secrets;print(secrets.token_hex(24))"; }

id ragsvc >/dev/null 2>&1 || useradd --system --home-dir /var/lib/ragsvc --create-home --shell /usr/sbin/nologin ragsvc
mkdir -p /opt/qdrant /var/lib/qdrant/storage /var/lib/qdrant/snapshots /etc/local-rag
install -m 0755 "$APP/vendor/qdrant" /opt/qdrant/qdrant

# .env with generated secrets (only replaces placeholders; idempotent)
[ -f "$APP/.env" ] || cp "$APP/.env.example" "$APP/.env"
grep -q CHANGE_ME_QDRANT_KEY "$APP/.env" && sed -i "s|CHANGE_ME_QDRANT_KEY|$(rnd)|" "$APP/.env"
grep -q CHANGE_ME_MCP_TOKEN "$APP/.env" && sed -i "s|CHANGE_ME_MCP_TOKEN|$(rnd)|" "$APP/.env"
QKEY="$(grep '^QDRANT_API_KEY=' "$APP/.env" | head -1 | cut -d= -f2- | cut -d' ' -f1)"
sed "s|__QDRANT_KEY__|$QKEY|" "$APP/deploy/qdrant.env.example" > /etc/local-rag/qdrant.env
chmod 600 /etc/local-rag/qdrant.env "$APP/.env"

# python env (offline)
[ -x "$APP/.venv/bin/python" ] || bash "$APP/scripts/offline_install.sh"

mkdir -p "$APP/data" "$APP/state"
chown -R ragsvc:ragsvc "$APP" /var/lib/qdrant /var/lib/ragsvc
chown root:root /etc/local-rag/qdrant.env && chmod 600 /etc/local-rag/qdrant.env  # systemd reads it as root
chmod 600 "$APP/.env"

for u in qdrant.service local-rag-mcp.service local-rag-sync.service local-rag-sync.timer \
         local-rag-full.service local-rag-full.timer; do
  install -m 0644 "$APP/deploy/$u" "/etc/systemd/system/$u"
done
systemctl daemon-reload
systemctl enable --now qdrant.service
systemctl enable local-rag-mcp.service local-rag-sync.timer local-rag-full.timer

echo
echo "Installed. NEXT STEPS:"
echo " 1. Edit $APP/.env : LITELLM_* , RAG_EMBED_MODEL, CONFLUENCE_* (and MCP_TLS_* or use nginx)."
echo " 2. cp $APP/confluence_roots.txt.example $APP/confluence_roots.txt  and list root page IDs."
echo " 3. sudo -u ragsvc $APP/.venv/bin/python $APP/scripts/smoke_test.py"
echo " 4. sudo -u ragsvc $APP/scripts/export_confluence.sh        # first export + index"
echo " 5. systemctl start local-rag-mcp local-rag-sync.timer local-rag-full.timer"
echo " 6. Open port ${MCP_PORT:-8765} (or 443 via nginx) to your client subnet only."
echo
echo "Your first MCP token: $(grep '^MCP_TOKENS=' "$APP/.env" | cut -d= -f2- | cut -d, -f1 | cut -d' ' -f1)"
