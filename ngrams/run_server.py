#!/usr/bin/env python3
"""
Convenience script to run the web dashboard.

Usage:
    python run_server.py
    python run_server.py --port 8080
    python run_server.py --debug
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from web.app import app

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run N-gram Model Web Dashboard')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print(f"""
╔═══════════════════════════════════════════════════════╗
║     N-gram Language Model Web Dashboard               ║
╠═══════════════════════════════════════════════════════╣
║  Open http://{args.host}:{args.port} in your browser         ║
╚═══════════════════════════════════════════════════════╝
    """)
    
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
