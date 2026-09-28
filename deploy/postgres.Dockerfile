FROM pgvector/pgvector:0.8.6-pg17-bookworm@sha256:cf134a767f474095eeba57e0117be8e568e011a63f33fbf252f14c9b760f8e6f AS package

USER root
ARG TARGETARCH
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl unzip; \
    case "$TARGETARCH" in \
      arm64) expected='c084c942caa9d6e35a84aaff8b21e6c51afa4126030aabf1d5e76f03f4f2a320' ;; \
      amd64) expected='93dbb144b09675ce5294d2a8655ed6b7f53a79cb7ebee1b7c8c3c148561a0383' ;; \
      *) exit 1 ;; \
    esac; \
    curl --fail --location --show-error --silent "https://github.com/timescale/pg_textsearch/releases/download/v1.4.0/pg-textsearch-v1.4.0-pg17-${TARGETARCH}.zip" --output /tmp/pg-textsearch.zip; \
    echo "$expected  /tmp/pg-textsearch.zip" | sha256sum --check --status; \
    unzip -p /tmp/pg-textsearch.zip > /tmp/pg-textsearch.deb; \
    test "$(dpkg-deb --field /tmp/pg-textsearch.deb Package)" = pg-textsearch-postgresql-17; \
    test "$(dpkg-deb --field /tmp/pg-textsearch.deb Version)" = 1.4.0-1; \
    test "$(dpkg-deb --field /tmp/pg-textsearch.deb Architecture)" = "$TARGETARCH"; \
    dpkg-deb --extract /tmp/pg-textsearch.deb /extension; \
    mkdir -p /extension/usr/share/doc/pg_textsearch; \
    curl --fail --location --show-error --silent https://raw.githubusercontent.com/timescale/pg_textsearch/v1.4.0/LICENSE --output /extension/usr/share/doc/pg_textsearch/LICENSE; \
    echo 'd33de21a123ce25b41722a5d10750984cb9c844c4d9b01add9e1b31f3ff452e5  /extension/usr/share/doc/pg_textsearch/LICENSE' | sha256sum --check --status

FROM pgvector/pgvector:0.8.6-pg17-bookworm@sha256:cf134a767f474095eeba57e0117be8e568e011a63f33fbf252f14c9b760f8e6f

COPY --from=package /extension/usr/lib/postgresql/17/lib/pg_textsearch.so /usr/lib/postgresql/17/lib/pg_textsearch.so
COPY --from=package /extension/usr/share/postgresql/17/extension/ /usr/share/postgresql/17/extension/
COPY --from=package /extension/usr/share/doc/pg_textsearch/LICENSE /usr/share/doc/pg_textsearch/LICENSE
