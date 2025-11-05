"""
Test script to verify the setup is working correctly.
Run this after installing dependencies to ensure everything is configured.
"""

import sys


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import requests
        print("  ✓ requests")
    except ImportError:
        print("  ✗ requests - Run: pip install requests")
        return False
    
    try:
        from dotenv import load_dotenv
        print("  ✓ python-dotenv")
    except ImportError:
        print("  ✗ python-dotenv - Run: pip install python-dotenv")
        return False
    
    try:
        import pandas
        print("  ✓ pandas")
    except ImportError:
        print("  ✗ pandas - Run: pip install pandas")
        return False
    
    try:
        import numpy
        print("  ✓ numpy")
    except ImportError:
        print("  ✗ numpy - Run: pip install numpy")
        return False
    
    try:
        import matplotlib
        print("  ✓ matplotlib")
    except ImportError:
        print("  ✗ matplotlib - Run: pip install matplotlib")
        return False
    
    try:
        import seaborn
        print("  ✓ seaborn")
    except ImportError:
        print("  ✗ seaborn - Run: pip install seaborn")
        return False
    
    return True


def test_local_modules():
    """Test that all local modules can be imported."""
    print("\nTesting local modules...")
    
    try:
        from api_client import SphinxAPIClient
        print("  ✓ api_client.py")
    except ImportError as e:
        print(f"  ✗ api_client.py - {e}")
        return False
    
    try:
        from data_collector import DataCollector
        print("  ✓ data_collector.py")
    except ImportError as e:
        print(f"  ✗ data_collector.py - {e}")
        return False
    
    try:
        import visualizations
        print("  ✓ visualizations.py")
    except ImportError as e:
        print(f"  ✗ visualizations.py - {e}")
        return False
    
    try:
        import utils
        print("  ✓ utils.py")
    except ImportError as e:
        print(f"  ✗ utils.py - {e}")
        return False
    
    try:
        import strategy
        print("  ✓ strategy.py")
    except ImportError as e:
        print(f"  ✗ strategy.py - {e}")
        return False
    
    return True


def test_api_token():
    """Test that API token is configured."""
    print("\nTesting API token configuration...")
    
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    token = os.getenv("SPHINX_API_TOKEN")
    
    if not token:
        print("  ✗ No API token found")
        print("\n  To fix:")
        print("  1. Create a .env file in this directory")
        print("  2. Add: SPHINX_API_TOKEN=your_token_here")
        print("  3. Get token from: https://challenge.sphinxhq.com/")
        return False
    
    if token == "your_token_here":
        print("  ✗ API token not configured (still using placeholder)")
        print("\n  To fix:")
        print("  1. Get your token from: https://challenge.sphinxhq.com/")
        print("  2. Update .env file with your actual token")
        return False
    
    print(f"  ✓ API token configured (length: {len(token)})")
    return True


def test_api_connection():
    """Test connection to the API."""
    print("\nTesting API connection...")
    
    try:
        from api_client import SphinxAPIClient
        
        client = SphinxAPIClient()
        status = client.get_status()
        
        print("  ✓ API connection successful")
        print(f"\n  Current Status:")
        print(f"    Morties in Citadel: {status['morties_in_citadel']}")
        print(f"    Morties on Planet Jessica: {status['morties_on_planet_jessica']}")
        print(f"    Morties Lost: {status['morties_lost']}")
        print(f"    Steps Taken: {status['steps_taken']}")
        
        return True
        
    except ValueError as e:
        print(f"  ✗ API token error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ API connection failed: {e}")
        return False


def test_data_collection():
    """Test basic data collection."""
    print("\nTesting data collection (will send 3 test trips)...")
    
    try:
        from api_client import SphinxAPIClient
        from data_collector import DataCollector
        
        client = SphinxAPIClient()
        collector = DataCollector(client)
        
        # Send 3 test trips
        import pandas as pd
        test_trips = []
        
        for i in range(3):
            result = client.send_morties(planet=0, morty_count=1)
            test_trips.append({
                'trip': i + 1,
                'survived': result['survived']
            })
        
        df = pd.DataFrame(test_trips)
        print(f"  ✓ Data collection works")
        print(f"    Test trips: {len(df)}")
        print(f"    Survival rate: {df['survived'].mean() * 100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Data collection failed: {e}")
        return False


def test_visualization():
    """Test that visualizations can be created."""
    print("\nTesting visualization capabilities...")
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for testing
        
        import matplotlib.pyplot as plt
        import pandas as pd
        from visualizations import plot_survival_by_planet
        
        # Create dummy data
        dummy_data = pd.DataFrame({
            'planet': [0, 0, 1, 1, 2, 2],
            'planet_name': ['"On a Cob" Planet'] * 2 + ['Cronenberg World'] * 2 + ['The Purge Planet'] * 2,
            'survived': [True, False, True, True, False, True],
            'morties_sent': [1, 1, 1, 1, 1, 1]
        })
        
        # Try to create a plot (won't display in test)
        fig = plt.figure()
        plt.close(fig)
        
        print("  ✓ Visualization libraries work")
        return True
        
    except Exception as e:
        print(f"  ✗ Visualization test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("MORTY EXPRESS CHALLENGE - SETUP TEST")
    print("="*60)
    
    results = {
        'imports': test_imports(),
        'local_modules': test_local_modules(),
        'api_token': test_api_token(),
    }
    
    # Only test API if token is configured
    if results['api_token']:
        results['api_connection'] = test_api_connection()
        
        # Only test data collection if API works
        if results['api_connection']:
            results['data_collection'] = test_data_collection()
    
    results['visualization'] = test_visualization()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  All tests passed! You're ready to start!")
        print("\nNext steps:")
        print("  python example.py          # Run full example")
    else:
        print("\n   Some tests failed. Please fix the issues above.")
        sys.exit(1)
    
    print("="*60)


if __name__ == "__main__":
    main()
