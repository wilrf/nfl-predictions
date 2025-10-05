#!/usr/bin/env python3
"""
Test script to verify Vercel, GitHub CLI, and Playwright setup
"""
import subprocess
import sys
import os

def test_command(name, command, expected_in_output=None):
    """Test if a command works and optionally check output"""
    print(f"\n{'='*60}")
    print(f"Testing {name}...")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            print(f"✅ {name} is working!")
            if expected_in_output and expected_in_output in result.stdout:
                print(f"   Output contains: {expected_in_output}")
            print(f"   Output: {result.stdout.strip()[:200]}")
            return True
        else:
            print(f"❌ {name} failed with code {result.returncode}")
            print(f"   Error: {result.stderr.strip()[:200]}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏱️  {name} timed out")
        return False
    except Exception as e:
        print(f"❌ {name} error: {str(e)}")
        return False

def main():
    print("🚀 NFL Betting System - Tool Setup Verification")
    print("="*60)

    tests = [
        ("Vercel CLI", "vercel --version", "Vercel CLI"),
        ("GitHub CLI", "gh --version", "gh version"),
        ("Playwright", "python3 -m playwright --version", "Version"),
        ("Git", "git --version", "git version"),
        ("Python", "python3 --version", "Python 3"),
    ]

    results = {}
    for name, cmd, expected in tests:
        results[name] = test_command(name, cmd, expected)

    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")

    for tool, status in results.items():
        emoji = "✅" if status else "❌"
        print(f"{emoji} {tool}: {'READY' if status else 'FAILED'}")

    # Check project structure
    print(f"\n{'='*60}")
    print("📁 Project Structure Check")
    print(f"{'='*60}")

    required_paths = [
        "improved_nfl_system/web/app.py",
        "improved_nfl_system/main.py",
        "vercel.json",
        "playwright.config.py",
    ]

    for path in required_paths:
        full_path = os.path.join("/Users/wilfowler/Sports Model", path)
        exists = os.path.exists(full_path)
        emoji = "✅" if exists else "❌"
        print(f"{emoji} {path}")

    print(f"\n{'='*60}")
    print("🎯 Next Steps:")
    print(f"{'='*60}")
    print("1. Deploy to Vercel: vercel")
    print("2. Create GitHub repo: gh repo create")
    print("3. Run Playwright tests: python3 -m pytest improved_nfl_system/web/tests/")
    print("4. Start visual updates!")

    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
