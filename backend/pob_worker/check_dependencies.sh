#!/bin/bash
# Check all required dependencies for PoB worker

set -e

echo "Checking PoB Worker Dependencies..."
echo "===================================="
echo

# Check LuaJIT
echo -n "LuaJIT: "
if command -v luajit &> /dev/null; then
    VERSION=$(luajit -v 2>&1 | head -1)
    echo "✓ $VERSION"
else
    echo "✗ NOT FOUND"
    echo "  Install: sudo apt-get install luajit"
    exit 1
fi

# Check lua-cjson
echo -n "lua-cjson: "
if luajit -e "require('cjson')" &> /dev/null; then
    echo "✓ Installed"
else
    echo "✗ NOT FOUND"
    echo "  Install: sudo apt-get install lua-cjson"
    exit 1
fi

# Check lua-utf8
echo -n "lua-utf8: "
if luajit -e "require('lua-utf8')" &> /dev/null; then
    echo "✓ Installed"
else
    echo "✗ NOT FOUND"
    echo "  Install: sudo apt-get install lua-utf8"
    echo "    OR: sudo luarocks --lua-version=5.1 install luautf8"
    exit 1
fi

echo
echo "All dependencies satisfied!"
echo
echo "Next steps:"
echo "1. Test the worker: cd backend/pob_worker && ./test_worker.sh"
echo "2. Start ML loop: make ml-loop-start"
