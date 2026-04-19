"""
Webhook Testing Script - Uses Environment Variables
====================================================

Tests all 4 webhooks by reading from environment variables.
Same variables used by daily_signal_generator_hybrid.py

USAGE:
  python test_webhooks.py

ENVIRONMENT VARIABLES REQUIRED:
  OA_WEBHOOK_SPS_1X
  OA_WEBHOOK_SPS_2X
  OA_WEBHOOK_SCS_1X
  OA_WEBHOOK_SCS_2X
"""

import requests
import json
import os
import time
from datetime import datetime

# Delay between webhook tests (seconds)
WEBHOOK_DELAY = 60  # 1 minute between tests to avoid rate limiting

# Read webhooks from environment variables (same as production script)
WEBHOOKS = {
    "SPS": os.environ.get("OA_WEBHOOK_SPS"),
    "SCS": os.environ.get("OA_WEBHOOK_SCS"),
    "SPS_2X": os.environ.get("OA_WEBHOOK_SPS_2X"),
    "SCS_2X": os.environ.get("OA_WEBHOOK_SCS_2X"),
}

# Test payloads for each webhook type
TEST_SIGNALS = {
    "SPS": {
        "signal_type": "SPS",
        "direction": "BULLISH",
        "probability": 0.72,
        "threshold": 0.55,
        "confidence": "MEDIUM",
        "position_size": 1,
        "regime": "MEDIUM_VOL_BULL",
        "vix": 18.5,
        "trend": "BULL",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "spx_price": 6000.0,
        "td_count": 3,
        "td_direction": "bullish"
    },

    "SCS": {
        "signal_type": "SCS",
        "direction": "BEARISH",
        "probability": 0.68,
        "threshold": 0.60,
        "confidence": "LOW",
        "position_size": 1,
        "regime": "MEDIUM_VOL_BULL",
        "vix": 19.0,
        "trend": "BULL",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "spx_price": 6000.0,
        "td_count": 4,
        "td_direction": "bearish"
    },

    # 2X payloads (same signal but double position)
    "SPS_2X": {
        "signal_type": "SPS",
        "direction": "BULLISH",
        "probability": 0.72,
        "threshold": 0.55,
        "confidence": "MEDIUM",
        "position_size": 2,
        "regime": "MEDIUM_VOL_BULL",
        "vix": 18.5,
        "trend": "BULL",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "spx_price": 6000.0,
        "td_count": 3,
        "td_direction": "bullish"
    },

    "SCS_2X": {
        "signal_type": "SCS",
        "direction": "BEARISH",
        "probability": 0.68,
        "threshold": 0.60,
        "confidence": "LOW",
        "position_size": 2,
        "regime": "MEDIUM_VOL_BULL",
        "vix": 19.0,
        "trend": "BULL",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "spx_price": 6000.0,
        "td_count": 4,
        "td_direction": "bearish"
    }
}


def test_webhook(name, url, payload):
    """Test a single webhook"""
    
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"{'='*80}")
    
    if not url:
        print(f"❌ SKIPPED: Environment variable OA_WEBHOOK_{name} not set")
        return False
    
    print(f"URL: {url[:50]}..." if len(url) > 50 else f"URL: {url}")
    print(f"\nPayload:")
    print(json.dumps(payload, indent=2))
    
    try:
        print(f"\n🚀 Sending webhook...")
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ SUCCESS: HTTP {response.status_code}")
            print(f"Response: {response.text[:200]}" if len(response.text) > 200 else f"Response: {response.text}")
            return True
        else:
            print(f"⚠️  WARNING: HTTP {response.status_code}")
            print(f"Response: {response.text[:200]}" if len(response.text) > 200 else f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ ERROR: Request timeout (>10 seconds)")
        return False
    except requests.exceptions.ConnectionError:
        print(f"❌ ERROR: Connection failed - check URL")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    print("="*80)
    print("WEBHOOK TESTING SCRIPT")
    print("="*80)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check which webhooks are configured
    configured = {k: v for k, v in WEBHOOKS.items() if v}
    missing = {k: v for k, v in WEBHOOKS.items() if not v}
    
    print(f"\n📋 Configuration Status:")
    print(f"   Configured: {len(configured)}/4")
    print(f"   Missing: {len(missing)}/4")
    
    if configured:
        print(f"\n✅ Configured webhooks:")
        for name in configured.keys():
            print(f"   - {name}")
    
    if missing:
        print(f"\n⚠️  Missing webhooks:")
        for name in missing.keys():
            print(f"   - {name} (set OA_WEBHOOK_{name})")
    
    if not configured:
        print(f"\n❌ No webhooks configured. Set environment variables and try again.")
        print(f"\nExample:")
        print(f"  export OA_WEBHOOK_SPS_1X='https://webhooks.optionalpha.com/...'")
        return
    
    # Ask user which webhooks to test
    print(f"\n{'='*80}")
    print(f"TEST OPTIONS")
    print(f"{'='*80}")
    print(f"1. Test all configured webhooks (with {WEBHOOK_DELAY}s delay)")
    print(f"2. Test specific webhook (no delay)")
    print(f"3. Test all with custom delay")
    print(f"4. Exit")
    
    choice = input(f"\nChoice (1-4): ").strip()
    
    results = {}
    delay = WEBHOOK_DELAY
    
    if choice == "1":
        # Test all configured webhooks with default delay
        print(f"\n🧪 Testing all configured webhooks...")
        print(f"⏱️  Delay between tests: {delay} seconds")
        
        webhook_list = list(configured.items())
        for i, (name, url) in enumerate(webhook_list):
            payload = TEST_SIGNALS[name]
            success = test_webhook(name, url, payload)
            results[name] = success
            
            # Wait between webhooks (except after the last one)
            if i < len(webhook_list) - 1:
                print(f"\n⏳ Waiting {delay} seconds before next test...")
                for remaining in range(delay, 0, -1):
                    print(f"   {remaining} seconds remaining...", end='\r')
                    time.sleep(1)
                print()  # New line after countdown
    
    elif choice == "2":
        # Test specific webhook (no delay)
        print(f"\nAvailable webhooks:")
        webhook_list = list(configured.keys())
        for i, name in enumerate(webhook_list, 1):
            print(f"{i}. {name}")
        
        webhook_choice = input(f"\nSelect webhook (1-{len(webhook_list)}): ").strip()
        
        try:
            idx = int(webhook_choice) - 1
            if 0 <= idx < len(webhook_list):
                name = webhook_list[idx]
                url = configured[name]
                payload = TEST_SIGNALS[name]
                success = test_webhook(name, url, payload)
                results[name] = success
            else:
                print(f"❌ Invalid choice")
                return
        except ValueError:
            print(f"❌ Invalid input")
            return
    
    elif choice == "3":
        # Test all with custom delay
        custom_delay_input = input(f"\nEnter delay in seconds (default {WEBHOOK_DELAY}): ").strip()
        
        try:
            if custom_delay_input:
                delay = int(custom_delay_input)
            else:
                delay = WEBHOOK_DELAY
            
            if delay < 0:
                print(f"❌ Delay must be positive")
                return
                
        except ValueError:
            print(f"❌ Invalid delay value")
            return
        
        print(f"\n🧪 Testing all configured webhooks...")
        print(f"⏱️  Delay between tests: {delay} seconds")
        
        webhook_list = list(configured.items())
        for i, (name, url) in enumerate(webhook_list):
            payload = TEST_SIGNALS[name]
            success = test_webhook(name, url, payload)
            results[name] = success
            
            # Wait between webhooks (except after the last one)
            if i < len(webhook_list) - 1 and delay > 0:
                print(f"\n⏳ Waiting {delay} seconds before next test...")
                for remaining in range(delay, 0, -1):
                    print(f"   {remaining} seconds remaining...", end='\r')
                    time.sleep(1)
                print()  # New line after countdown
    
    else:
        print(f"Exiting...")
        return
    
    # Summary
    if results:
        print(f"\n{'='*80}")
        print(f"TEST RESULTS SUMMARY")
        print(f"{'='*80}")
        
        passed = sum(1 for v in results.values() if v)
        failed = sum(1 for v in results.values() if not v)
        
        print(f"\n✅ Passed: {passed}/{len(results)}")
        print(f"❌ Failed: {failed}/{len(results)}")
        
        if failed > 0:
            print(f"\n⚠️  Failed webhooks:")
            for name, success in results.items():
                if not success:
                    print(f"   - {name}")
        
        if passed == len(results):
            print(f"\n🎉 All webhooks working correctly!")
    
    print(f"\n{'='*80}")
    print(f"TESTING COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
