"""
FastAPI web application for NFL suggestions
FAIL FAST: Any startup error prevents launch
"""
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import logging
import sys
import traceback
from typing import Dict, List

from web.bridge.nfl_bridge import NFLSystemBridge, BridgeError
from web.config.web_config import WebConfig, ConfigError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Validate configuration on startup
try:
    WebConfig.validate()
    logger.info("✅ Configuration validated")
except ConfigError as e:
    logger.error(f"❌ Configuration error: {e}")
    sys.exit(1)

# Initialize FastAPI
app = FastAPI(
    title="NFL Betting Suggestions",
    description="Real-time NFL betting suggestions with confidence scoring",
    version="1.0.0"
)

# Static files and templates
app.mount("/static", StaticFiles(directory=WebConfig.STATIC_DIR), name="static")
templates = Jinja2Templates(directory=WebConfig.TEMPLATES_DIR)

# Global bridge instance
bridge = None


@app.on_event("startup")
async def startup_event():
    """Initialize NFL system bridge on startup"""
    global bridge
    try:
        bridge = NFLSystemBridge()
        logger.info("✅ NFL System Bridge ready")
    except Exception as e:
        logger.error(f"❌ Failed to initialize bridge: {e}")
        logger.error(traceback.format_exc())
        # FAIL FAST - don't start if bridge fails
        raise e


def get_bridge() -> NFLSystemBridge:
    """Dependency to get bridge instance"""
    if bridge is None:
        raise HTTPException(500, "System not initialized")
    return bridge


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    try:
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "config": WebConfig
        })
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        raise HTTPException(500, "Dashboard unavailable")


@app.get("/api/suggestions")
async def get_suggestions(bridge: NFLSystemBridge = Depends(get_bridge)):
    """Get current suggestions as HTML fragment for HTMX"""
    try:
        suggestions = bridge.get_current_suggestions()

        # Group by confidence tiers
        tiers = {
            'premium': [s for s in suggestions if s['confidence'] >= WebConfig.CONFIDENCE_TIERS['premium']],
            'standard': [s for s in suggestions if WebConfig.CONFIDENCE_TIERS['standard'] <= s['confidence'] < WebConfig.CONFIDENCE_TIERS['premium']],
            'reference': [s for s in suggestions if WebConfig.CONFIDENCE_TIERS['reference'] <= s['confidence'] < WebConfig.CONFIDENCE_TIERS['standard']]
        }

        # Render HTML fragment
        html = render_suggestions_tiers(tiers)
        return HTMLResponse(html)

    except BridgeError as e:
        # Safe error for users
        logger.warning(f"Bridge error in suggestions API: {e}")
        return HTMLResponse(f'<div class="text-red-400 p-4 text-center">⚠️ {str(e)}</div>')
    except Exception as e:
        logger.error(f"Suggestions API error: {e}")
        logger.error(traceback.format_exc())
        return HTMLResponse('<div class="text-red-400 p-4 text-center">⚠️ Service temporarily unavailable</div>')


@app.get("/api/performance")
async def get_performance(bridge: NFLSystemBridge = Depends(get_bridge)):
    """Get performance data for charts"""
    try:
        # Get CLV data from database
        clv_data = bridge.get_clv_performance()
        return {"clv_data": clv_data, "status": "success"}
    except Exception as e:
        logger.error(f"Performance API error: {e}")
        return {"error": "Performance data unavailable", "status": "error"}


@app.get("/api/status")
async def get_status(bridge: NFLSystemBridge = Depends(get_bridge)):
    """Get system status information"""
    try:
        status = bridge.get_system_status()
        return status
    except Exception as e:
        logger.error(f"Status API error: {e}")
        return {"status": "error", "message": "Status unavailable"}


def render_suggestions_tiers(tiers: Dict) -> str:
    """Render tiered suggestions as HTML"""
    html_parts = []

    tier_configs = [
        ('premium', WebConfig.get_tier_config('premium')),
        ('standard', WebConfig.get_tier_config('standard')),
        ('reference', WebConfig.get_tier_config('reference'))
    ]

    for tier_key, tier_config in tier_configs:
        suggestions = tiers.get(tier_key, [])
        if suggestions:
            html_parts.append(f'''
            <div class="mb-6">
                <h3 class="text-xl font-bold mb-3 {tier_config['text_color']}">{tier_config['title']}</h3>
                <div class="space-y-3">
            ''')

            for suggestion in suggestions:
                html_parts.append(render_suggestion_card(suggestion, tier_config))

            html_parts.append('</div></div>')

    if not any(tiers.values()):
        html_parts.append('''
            <div class="text-gray-400 p-8 text-center">
                <div class="text-6xl mb-4">🏈</div>
                <h3 class="text-xl font-semibold mb-2">No Suggestions Available</h3>
                <p>No games meet the minimum edge requirement for current week.</p>
                <p class="text-sm mt-2">Check back when new games are available.</p>
            </div>
        ''')

    return ''.join(html_parts)


def render_suggestion_card(suggestion: Dict, tier_config: Dict) -> str:
    """Render individual suggestion card"""
    # Correlation warnings
    warnings_html = ""
    if suggestion.get('correlation_warnings'):
        warnings_html = f'''
        <div class="mt-2 p-2 bg-yellow-900/50 rounded text-yellow-200 text-sm">
            ⚠️ {suggestion['correlation_warnings'][0]}
        </div>
        '''

    # Game time display
    game_time = suggestion.get('game_time', 'TBD')
    if isinstance(game_time, str) and game_time != 'TBD':
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(game_time.replace('Z', '+00:00'))
            game_time_display = dt.strftime('%a %I:%M %p')
        except:
            game_time_display = game_time
    else:
        game_time_display = 'TBD'

    return f'''
    <div class="bg-gray-800 border-l-4 {tier_config['border_color']} {tier_config['bg_color']} p-4 rounded-r hover:bg-gray-700 transition-colors">
        <div class="flex justify-between items-start">
            <div class="flex-1">
                <h4 class="font-bold text-lg">{suggestion['team']} {suggestion['bet_type'].upper()}</h4>
                <p class="text-gray-300">{suggestion['line_display']} @ {suggestion['odds']}</p>
                <p class="text-gray-400 text-sm">{game_time_display}</p>
            </div>
            <div class="text-right ml-4">
                <div class="text-2xl font-bold {tier_config['text_color']}">{suggestion['confidence']}</div>
                <div class="text-sm text-gray-400">confidence</div>
            </div>
        </div>

        <div class="flex justify-between mt-3 text-sm text-gray-300">
            <span>Edge: <span class="font-semibold">{suggestion['edge']:.1%}</span></span>
            <span>Margin: <span class="font-semibold">{suggestion['margin']}</span></span>
            <span>Kelly: <span class="font-semibold">{suggestion['kelly_fraction']:.1%}</span></span>
        </div>
        {warnings_html}
    </div>
    '''


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Custom 404 page"""
    return templates.TemplateResponse(
        "error.html",
        {"request": request, "error": "Page not found", "code": 404},
        status_code=404
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    """Custom 500 page"""
    logger.error(f"Internal server error: {exc}")
    return templates.TemplateResponse(
        "error.html",
        {"request": request, "error": "Internal server error", "code": 500},
        status_code=500
    )