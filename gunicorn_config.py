"""Gunicorn config for Render deployment."""
import os

# Render sets PORT (default 10000). Must bind to 0.0.0.0 for Render to detect.
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
workers = 1
threads = 2
timeout = 300
worker_timeout = 300
keepalive = 5
