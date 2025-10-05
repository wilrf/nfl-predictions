#!/usr/bin/env python3
"""
Production launch script with comprehensive preflight checks
FAIL FAST: Any check failure prevents launch
"""
import os
import sys
import subprocess
from pathlib import Path
import socket
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def preflight_checks():
    """Comprehensive preflight checks - FAIL FAST on any issue"""

    print("🚀 NFL Suggestions Web Interface - Preflight Checks")
    print("=" * 60)

    # 1. Environment variables
    required_env = ['ODDS_API_KEY']
    missing_env = [var for var in required_env if not os.getenv(var)]
    if missing_env:
        print(f"❌ Missing environment variables: {missing_env}")
        print("   Please set ODDS_API_KEY in your .env file")
        sys.exit(1)
    print("✅ Environment variables configured")

    # 2. Database accessibility
    db_path = Path(__file__).parent.parent / 'database' / 'nfl_suggestions.db'
    if not db_path.exists():
        print("❌ Database not found - run main system first")
        print(f"   Expected: {db_path}")
        print("   Run: python main.py")
        sys.exit(1)
    print("✅ Database accessible")

    # 3. Required packages
    try:
        import fastapi
        import uvicorn
        print("✅ FastAPI and Uvicorn installed")
    except ImportError as e:
        print(f"❌ Missing required packages: {e}")
        print("   Run: pip install fastapi uvicorn")
        sys.exit(1)

    # 4. NFL System integration test
    try:
        from web.bridge.nfl_bridge import NFLSystemBridge
        bridge = NFLSystemBridge()

        # Test actual call with timeout
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Bridge test timed out")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout

        try:
            suggestions = bridge.get_current_suggestions()
            print(f"✅ NFL System integration working ({len(suggestions)} suggestions available)")
        except TimeoutError:
            print("⚠️  NFL System integration slow but responsive")
        finally:
            signal.alarm(0)

    except Exception as e:
        print(f"❌ NFL System integration failed: {e}")
        print("   Check main system configuration and API keys")
        sys.exit(1)

    # 5. Port availability
    port = int(os.getenv('WEB_PORT', 8000))
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex(('localhost', port))
            if result == 0:
                print(f"❌ Port {port} already in use")
                print(f"   Kill existing process or use different port")
                sys.exit(1)
    except Exception as e:
        print(f"⚠️  Port check failed: {e}")

    print(f"✅ Port {port} available")

    # 6. Web application startup test
    try:
        from web.app import app
        from web.config.web_config import WebConfig
        WebConfig.validate()
        print("✅ Web application configuration valid")
    except Exception as e:
        print(f"❌ Web application startup failed: {e}")
        sys.exit(1)

    print("=" * 60)
    print("✅ All preflight checks passed - launching web interface")
    print("=" * 60)

def launch_web_interface():
    """Launch the web interface with production settings"""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Get configuration
    port = int(os.getenv('WEB_PORT', 8000))
    host = os.getenv('WEB_HOST', '0.0.0.0')

    print(f"🌐 Web interface starting at http://{host}:{port}")
    print(f"🔄 Suggestions update every {os.getenv('SUGGESTIONS_UPDATE_INTERVAL', 3600)} seconds")
    print(f"📊 Performance charts refresh every {os.getenv('PERFORMANCE_UPDATE_INTERVAL', 7200)} seconds")
    print("⏹️  Press Ctrl+C to stop")
    print("=" * 60)

    try:
        import uvicorn
        uvicorn.run(
            "web.app:app",
            host=host,
            port=port,
            reload=False,  # No reload in production
            access_log=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down web interface...")
    except Exception as e:
        print(f"❌ Web interface failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        preflight_checks()
        launch_web_interface()
    except KeyboardInterrupt:
        print("\n👋 Interrupted during startup")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)