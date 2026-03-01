# Path of Building LuaJIT Worker

Headless Path of Building calculation worker using JSON-RPC NDJSON protocol.

## Architecture

The worker implements a simple JSON-RPC protocol over stdin/stdout:

**Input** (NDJSON): One JSON request per line
```json
{"id": 1, "method": "evaluate", "params": {"xml": "<PathOfBuilding>...</PathOfBuilding>", "scenario_id": "...", "profile_id": "...", "ruleset_id": "..."}}
```

**Output** (NDJSON): One JSON response per line  
```json
{"id": 1, "result": {"metrics": {...}, "defense": {...}, "resources": {...}, ...}}
```

## Files

- `worker.lua` - Main LuaJIT worker script
- `Dockerfile` - Docker image for running worker in container
- `test_worker.sh` - Simple bash test script
- `README.md` - This file

## Running Locally

### Requirements

- LuaJIT 2.1+
- lua-cjson (JSON encoding/decoding)
- lua-filesystem (optional, for file operations)

### Installation (Ubuntu/Debian)

```bash
sudo apt-get install -y luajit lua-cjson lua-filesystem
```

### Test

```bash
cd PathOfBuilding/worker
./test_worker.sh
```

## Running with Docker

### Build Image

```bash
docker compose build pob-worker
```

### Test with Docker

```bash
# Read test build XML
TEST_XML=$(cat PathOfBuilding/spec/TestBuilds/3.13/OccVortex.xml)

# Create JSON-RPC request (escape quotes and remove newlines)
REQUEST='{"id":1,"method":"evaluate","params":{"xml":"'"$(echo "$TEST_XML" | sed 's/"/\\"/g' | tr -d '\n')"'","scenario_id":"test"}}'

# Run worker
echo "$REQUEST" | docker run -i --rm pob-worker:latest
```

## Integration with worker_pool.py

The worker is designed to be spawned by `backend/engine/worker_pool.py`.

### Local Mode
```python
# backend/app/settings.py
POB_WORKER_CMD = "luajit"
POB_WORKER_ARGS = ["/path/to/PathOfBuilding/worker/worker.lua"]
```

### Docker Mode
```python
# backend/app/settings.py
POB_WORKER_CMD = "docker"
POB_WORKER_ARGS = ["run", "-i", "--rm", "pob-worker:latest"]
```

## Output Format

The worker extracts the following metrics from PoB calculations:

### Metrics
- `full_dps` - Total DPS
- `max_hit` - Maximum hit survivability (using EHP as proxy)
- `utility_score` - Utility score (placeholder)

### Defense
- `armour` - Total armour
- `evasion` - Total evasion
- `resists` - Fire/Cold/Lightning/Chaos resistances

### Resources
- `life` - Total life
- `mana` - Total mana

### Reservation
- `reserved_percent` - Mana reservation percentage
- `available_percent` - Available mana percentage

### Attributes
- `strength`, `dexterity`, `intelligence` - Character attributes

## Error Handling

The worker returns JSON-RPC error responses for:

- Parse errors: `{"id": null, "error": {"code": -32700, "message": "Parse error"}}`
- Method not found: `{"id": 1, "error": {"code": -32601, "message": "Method not found"}}`
- Invalid params: `{"id": 1, "error": {"code": -32602, "message": "Invalid params"}}`
- Internal errors: `{"id": 1, "error": {"code": -32000, "message": "..."}}`

## Performance

- Worker startup: ~100-200ms (loading PoB modules)
- Per-build evaluation: ~50-500ms (depending on build complexity)
- Memory: ~50-100MB per worker process

## Troubleshooting

### "module 'cjson' not found"
Install lua-cjson: `sudo apt-get install lua-cjson`

### "module 'HeadlessWrapper' not found"  
Ensure working directory is `PathOfBuilding/worker/` and `package.path` includes `../src/?.lua`

### "build calculation output not available"
The build XML may be invalid or PoB couldn't calculate stats. Check stderr for PoB errors.

### Permission denied (Docker)
Add user to docker group: `sudo usermod -aG docker $USER` then logout/login
