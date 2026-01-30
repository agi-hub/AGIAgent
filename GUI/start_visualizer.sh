#!/bin/bash
# Convenience script to start the Agent Status Visualizer

# Default values
OUTPUT_DIR=""
HOST="0.0.0.0"
PORT=5000
DEBUG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [-d|--output-dir <path>] [--host <host>] [--port <port>] [--debug]"
            echo ""
            echo "Options:"
            echo "  -d, --output-dir <path>  Path to output directory (optional, auto-detects latest if not provided)"
            echo "  --host <host>            Host to bind (default: 0.0.0.0, accessible from all network interfaces)"
            echo "  --port <port>            Port to bind (default: 5000)"
            echo "  --debug                  Enable debug mode"
            echo ""
            echo "Examples:"
            echo "  $0 -d output_20251211_105151"
            echo "  $0 --output-dir output_20251211_105151"
            echo "  $0  # Auto-detects latest output_* directory"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Output directory is now optional - Python script will auto-detect if not provided

# Check if output directory exists (only if provided)
if [ -n "$OUTPUT_DIR" ] && [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory does not exist: $OUTPUT_DIR"
    exit 1
fi

# Check if Python script exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/agent_status_visualizer.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Start the visualizer
echo "Starting Agent Status Visualizer..."
if [ -n "$OUTPUT_DIR" ]; then
    echo "Output directory: $OUTPUT_DIR"
else
    echo "Output directory: Auto-detecting latest output_* directory..."
fi
echo "Server URL: http://$HOST:$PORT"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

if [ -n "$OUTPUT_DIR" ]; then
    python3 "$PYTHON_SCRIPT" --output-dir "$OUTPUT_DIR" --host "$HOST" --port "$PORT" $DEBUG
else
    python3 "$PYTHON_SCRIPT" --host "$HOST" --port "$PORT" $DEBUG
fi

