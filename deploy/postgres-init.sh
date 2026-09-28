#!/bin/sh
set -eu
psql --username "$POSTGRES_USER" --dbname postgres --set ON_ERROR_STOP=1 \
  --set app_password="$PFL_DB_PASSWORD" \
  --set read_password="$PFL_DB_READ_PASSWORD" \
  --set langfuse_password="$LANGFUSE_DB_PASSWORD" <<'SQL'
CREATE ROLE assistant LOGIN PASSWORD :'app_password';
CREATE ROLE assistant_readonly LOGIN PASSWORD :'read_password';
CREATE ROLE langfuse LOGIN PASSWORD :'langfuse_password';
CREATE DATABASE assistant OWNER assistant;
CREATE DATABASE langfuse OWNER langfuse;
\connect assistant
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_textsearch;
GRANT CONNECT ON DATABASE assistant TO assistant_readonly;
GRANT USAGE ON SCHEMA public TO assistant_readonly;
SQL
