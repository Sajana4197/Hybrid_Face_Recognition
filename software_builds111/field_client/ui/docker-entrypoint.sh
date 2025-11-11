#!/bin/sh
set -e

# Generate runtime config for the SPA
echo "window.APP_CONFIG = { apiBaseUrl: \"${API_BASE_URL}\" };" > /usr/share/nginx/html/config.js

exec "$@"