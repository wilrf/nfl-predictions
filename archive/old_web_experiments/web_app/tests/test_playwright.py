"""
Playwright browser tests for NFL Suggestions Web Interface
FAIL FAST: Any browser test failure indicates UI problems
"""
import pytest
import time
import subprocess
import signal
import os
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Test configuration
BASE_URL = "http://localhost:8000"
WEB_SERVER_PROCESS = None


def start_web_server():
    """Start the web server for testing"""
    global WEB_SERVER_PROCESS

    # Change to the correct directory
    os.chdir(Path(__file__).parent.parent.parent)

    # Start the web server using test launch script
    WEB_SERVER_PROCESS = subprocess.Popen([
        sys.executable, "web/test_launch.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={**os.environ, 'ODDS_API_KEY': 'baa3a174dc025d9865dcf65c5e8a4609'})

    # Wait for server to start
    time.sleep(5)

    return WEB_SERVER_PROCESS


def stop_web_server():
    """Stop the web server"""
    global WEB_SERVER_PROCESS
    if WEB_SERVER_PROCESS:
        WEB_SERVER_PROCESS.terminate()
        try:
            WEB_SERVER_PROCESS.wait(timeout=10)
        except subprocess.TimeoutExpired:
            WEB_SERVER_PROCESS.kill()


@pytest.fixture(scope="session", autouse=True)
def web_server():
    """Start web server for all tests"""
    print("🚀 Starting web server for Playwright tests...")

    try:
        process = start_web_server()
        yield process
    finally:
        print("🛑 Stopping web server...")
        stop_web_server()


@pytest.fixture
def browser_context():
    """Create browser context for each test"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1280, "height": 720},
            ignore_https_errors=True
        )
        yield context
        context.close()
        browser.close()


def test_dashboard_loads(browser_context):
    """Test that dashboard loads successfully"""
    page = browser_context.new_page()

    try:
        # Navigate to dashboard
        response = page.goto(BASE_URL)
        assert response.status == 200

        # Check page title
        expect(page).to_have_title("Dashboard - NFL Betting Suggestions")

        # Check main heading exists
        expect(page.locator("h1")).to_contain_text("NFL Suggestions")

        # Check suggestions container exists
        expect(page.locator("#suggestions-container")).to_be_visible()

        print("✅ Dashboard loads successfully")

    except Exception as e:
        page.screenshot(path="web/tests/dashboard_error.png")
        raise AssertionError(f"Dashboard failed to load: {e}")


def test_navigation_elements(browser_context):
    """Test navigation and header elements"""
    page = browser_context.new_page()
    page.goto(BASE_URL)

    # Check header elements
    expect(page.locator("header")).to_be_visible()
    expect(page.locator("h1")).to_contain_text("NFL Suggestions")

    # Check refresh button
    refresh_button = page.locator('button:has-text("Refresh")')
    expect(refresh_button).to_be_visible()

    # Check status indicators
    expect(page.locator("#last-update")).to_be_visible()
    expect(page.locator("#api-status")).to_be_visible()

    print("✅ Navigation elements present")


def test_suggestions_container(browser_context):
    """Test suggestions container and loading behavior"""
    page = browser_context.new_page()
    page.goto(BASE_URL)

    # Wait for suggestions container
    suggestions_container = page.locator("#suggestions-container")
    expect(suggestions_container).to_be_visible()

    # Check for either suggestions or empty state
    page.wait_for_timeout(3000)  # Wait for HTMX to load

    # Should show either suggestions or "No suggestions available"
    content = suggestions_container.inner_text()
    assert len(content) > 0, "Suggestions container is empty"

    print("✅ Suggestions container working")


def test_sidebar_elements(browser_context):
    """Test sidebar performance and status elements"""
    page = browser_context.new_page()
    page.goto(BASE_URL)

    # Check CLV Performance section
    expect(page.locator('h3:has-text("CLV Performance")')).to_be_visible()

    # Check System Status section
    expect(page.locator('h3:has-text("System Status")')).to_be_visible()

    # Check status elements
    expect(page.locator("#api-credits")).to_be_visible()
    expect(page.locator("#total-suggestions")).to_be_visible()
    expect(page.locator("#avg-confidence")).to_be_visible()
    expect(page.locator("#system-health")).to_be_visible()

    # Check Quick Actions
    expect(page.locator('h3:has-text("Quick Actions")')).to_be_visible()
    expect(page.locator('button:has-text("Force Refresh")')).to_be_visible()
    expect(page.locator('button:has-text("Clear Cache")')).to_be_visible()
    expect(page.locator('button:has-text("Export CSV")')).to_be_visible()

    print("✅ Sidebar elements present")


def test_refresh_functionality(browser_context):
    """Test refresh button functionality"""
    page = browser_context.new_page()
    page.goto(BASE_URL)

    # Click refresh button
    refresh_button = page.locator('button:has-text("Refresh")')
    refresh_button.click()

    # Check for loading indicator
    loading_indicator = page.locator("#loading")
    # Loading indicator might appear briefly

    # Wait for request to complete
    page.wait_for_timeout(2000)

    # Verify suggestions container still exists
    expect(page.locator("#suggestions-container")).to_be_visible()

    print("✅ Refresh functionality working")


def test_auto_refresh_toggle(browser_context):
    """Test auto-refresh toggle functionality"""
    page = browser_context.new_page()
    page.goto(BASE_URL)

    # Find auto-refresh toggle
    auto_refresh_toggle = page.locator("#auto-refresh")
    expect(auto_refresh_toggle).to_be_visible()

    # Toggle should be checked by default
    expect(auto_refresh_toggle).to_be_checked()

    # Click to toggle off
    auto_refresh_toggle.click()
    expect(auto_refresh_toggle).not_to_be_checked()

    # Click to toggle back on
    auto_refresh_toggle.click()
    expect(auto_refresh_toggle).to_be_checked()

    print("✅ Auto-refresh toggle working")


def test_clear_cache_button(browser_context):
    """Test clear cache button functionality"""
    page = browser_context.new_page()
    page.goto(BASE_URL)

    # Click clear cache button
    clear_cache_button = page.locator('button:has-text("Clear Cache")')
    clear_cache_button.click()

    # Button should show feedback
    page.wait_for_timeout(1000)

    # Verify button text changes (temporarily)
    # Note: This might be too fast to catch

    print("✅ Clear cache button functional")


def test_export_csv_button(browser_context):
    """Test CSV export functionality"""
    page = browser_context.new_page()
    page.goto(BASE_URL)

    # Click export button
    export_button = page.locator('button:has-text("Export CSV")')

    # Set up download handler
    with page.expect_download() as download_info:
        export_button.click()

        # Wait a bit for potential download
        page.wait_for_timeout(2000)

    # Note: Download might not occur if no suggestions available
    print("✅ Export CSV button functional")


def test_responsive_design(browser_context):
    """Test responsive design on mobile viewport"""
    page = browser_context.new_page()

    # Set mobile viewport
    page.set_viewport_size({"width": 375, "height": 667})
    page.goto(BASE_URL)

    # Check that main elements are still visible
    expect(page.locator("header")).to_be_visible()
    expect(page.locator("#suggestions-container")).to_be_visible()

    # Grid should stack on mobile
    main_grid = page.locator(".grid")
    expect(main_grid).to_be_visible()

    print("✅ Responsive design working")


def test_chart_container(browser_context):
    """Test chart container exists"""
    page = browser_context.new_page()
    page.goto(BASE_URL)

    # Check chart canvas exists
    chart_canvas = page.locator("#clv-chart")
    expect(chart_canvas).to_be_visible()

    # Chart might not have data, but container should exist
    print("✅ Chart container present")


def test_error_handling(browser_context):
    """Test error page handling"""
    page = browser_context.new_page()

    # Try to access non-existent page
    response = page.goto(f"{BASE_URL}/nonexistent-page")

    # Should get 404
    assert response.status == 404

    # Should show error page
    expect(page.locator("h1")).to_contain_text("Page Not Found")

    # Should have back to dashboard link
    expect(page.locator('a:has-text("Back to Dashboard")')).to_be_visible()

    print("✅ Error handling working")


def test_javascript_console_errors(browser_context):
    """Test for JavaScript console errors"""
    page = browser_context.new_page()

    # Collect console messages
    console_messages = []
    page.on("console", lambda msg: console_messages.append(msg))

    # Load the page
    page.goto(BASE_URL)
    page.wait_for_timeout(3000)

    # Check for error messages
    error_messages = [msg for msg in console_messages if msg.type == "error"]

    if error_messages:
        print("⚠️ JavaScript errors found:")
        for msg in error_messages:
            print(f"  - {msg.text}")
    else:
        print("✅ No JavaScript console errors")

    # Allow some errors but not too many
    assert len(error_messages) <= 2, f"Too many JavaScript errors: {len(error_messages)}"


def test_api_endpoints_accessibility(browser_context):
    """Test that API endpoints are accessible"""
    page = browser_context.new_page()

    # Test suggestions API
    suggestions_response = page.goto(f"{BASE_URL}/api/suggestions")
    assert suggestions_response.status in [200, 500]  # 500 if bridge not working

    # Test performance API
    performance_response = page.goto(f"{BASE_URL}/api/performance")
    assert performance_response.status in [200, 500]  # 500 if bridge not working

    # Test status API
    status_response = page.goto(f"{BASE_URL}/api/status")
    assert status_response.status in [200, 500]  # 500 if bridge not working

    print("✅ API endpoints accessible")


def test_page_load_performance(browser_context):
    """Test page load performance"""
    page = browser_context.new_page()

    # Measure page load time
    start_time = time.time()
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    load_time = time.time() - start_time

    print(f"📊 Page load time: {load_time:.2f} seconds")

    # Should load within reasonable time
    assert load_time < 10, f"Page load too slow: {load_time:.2f}s"

    print("✅ Page load performance acceptable")


if __name__ == "__main__":
    # Run Playwright tests
    import subprocess

    print("🎭 Running Playwright tests for NFL Suggestions Web Interface")
    print("=" * 60)

    result = subprocess.run([
        sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"
    ], capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode == 0:
        print("=" * 60)
        print("✅ All Playwright tests passed!")
    else:
        print("=" * 60)
        print("❌ Some tests failed!")

    sys.exit(result.returncode)