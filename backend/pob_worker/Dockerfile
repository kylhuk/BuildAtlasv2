# Dockerfile for Path of Building LuaJIT Worker
# Implements headless PoB calculations with JSON-RPC NDJSON protocol

FROM alpine:3.19

# Install LuaJIT, Lua libraries, and dependencies
RUN apk add --no-cache \
    luajit \
    luajit-dev \
    lua5.1-cjson \
    lua5.1-filesystem \
    gcc \
    musl-dev \
    make \
    git \
    unzip \
    && rm -rf /var/cache/apk/*

# Create working directory
WORKDIR /app

# Copy PathOfBuilding source code
COPY src/ /app/src/
COPY runtime/ /app/runtime/
COPY manifest.xml /app/manifest.xml

# Copy worker script
COPY worker/worker.lua /app/worker/worker.lua

# Set working directory for worker
WORKDIR /app/worker

# Healthcheck: verify LuaJIT and cjson are available
RUN luajit -e "require('cjson'); print('LuaJIT worker ready')"

# Run worker
CMD ["luajit", "worker.lua"]
