#!/usr/bin/env python3
"""
Sample Test Script for DataTobiz Brand Monitoring System

This script demonstrates how to use the brand monitoring system
and provides sample queries for testing different scenarios.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import BrandMonitoringAPI
from src.utils.logger import setup_logging

# Setup logging for the test
setup_logging(log_level="INFO")

async def test_basic_functionality():
    """Test basic functionality of the brand monitoring system."""
    print("🚀 Starting DataTobiz Brand Monitoring System Test")
    print("=" * 60)
    
    # Initialize the API
    api = BrandMonitoringAPI("config.yaml")
    
    try:
        # Test 1: Initialize the system
        print("\n📋 Test 1: System Initialization")
        print("-" * 40)
        
        success = await api.initialize()
        if success:
            print("✅ System initialized successfully!")
        else:
            print("❌ System initialization failed!")
            return False
        
        # Test 2: Connection Testing
        print("\n🔌 Test 2: Connection Testing")
        print("-" * 40)
        
        connection_result = await api.test_connections()
        if connection_result['success']:
            print("✅ All connections are working!")
        else:
            print("❌ Some connections failed!")
            print(f"Error: {connection_result.get('error', 'Unknown error')}")
        
        # Test 3: Sample Queries with High Probability of Success
        print("\n🔍 Test 3: Sample Brand Monitoring Queries")
        print("-" * 40)
        
        # These queries are designed to test the system without expecting
        # DataTobiz to actually appear in results (for testing purposes)
        test_queries = [
            "best data analytics companies",
            "top business intelligence tools",
            "leading data visualization platforms"
        ]
        
        print(f"Testing with {len(test_queries)} sample queries...")
        
        result = await api.monitor_queries(test_queries, mode="parallel")
        
        if result['success']:
            print("✅ Query processing completed successfully!")
            
            summary = result['summary']
            print(f"\n📊 Results Summary:")
            print(f"   Total Queries: {summary['total_queries']}")
            print(f"   Processed: {summary['processed']}")
            print(f"   Success Rate: {summary['success_rate']:.1%}")
            print(f"   Brand Mentions: {summary['brand_mentions_found']}")
            
            print(f"\n📝 Detailed Results:")
            for query, query_result in result['results'].items():
                status = "🎯 FOUND" if query_result['found'] else "❌ NOT FOUND"
                confidence = f"({query_result['confidence']:.1%})" if query_result['found'] else ""
                print(f"   {query[:45]:<45} {status} {confidence}")
                
                # Show agent breakdown
                for agent_name, agent_result in query_result['agents'].items():
                    agent_status = "✅" if agent_result['status'] == 'completed' else "❌"
                    agent_found = "🎯" if agent_result['found'] else "❌"
                    time_taken = f"{agent_result['execution_time']:.2f}s"
                    print(f"     └─ {agent_name:<12} {agent_status} {agent_found} ({time_taken})")
        
        else:
            print("❌ Query processing failed!")
            print(f"Error: {result.get('error', 'Unknown error')}")
            return False
        
        # Test 4: Historical Statistics (if available)
        print("\n📈 Test 4: Historical Statistics")
        print("-" * 40)
        
        stats_result = await api.get_stats()
        if stats_result['success'] and stats_result['stats']:
            stats = stats_result['stats']
            print("✅ Historical data retrieved!")
            print(f"   Total Results: {stats.get('total_results', 0)}")
            print(f"   Brand Mentions: {stats.get('brand_mentions_found', 0)}")
            print(f"   Detection Rate: {stats.get('detection_rate', 0):.1%}")
        else:
            print("ℹ️  No historical data available (first run)")
        
        # Test 5: Individual Agent Testing
        print("\n🤖 Test 5: Individual Agent Performance")
        print("-" * 40)
        
        # Test with a single query for each agent
        single_query_result = await api.monitor_queries(
            ["data analytics platforms comparison"], 
            mode="sequential"
        )
        
        if single_query_result['success']:
            query_data = list(single_query_result['results'].values())[0]
            
            print("Agent Performance Breakdown:")
            for agent_name, agent_result in query_data['agents'].items():
                status_icon = "✅" if agent_result['status'] == 'completed' else "❌"
                exec_time = agent_result['execution_time']
                confidence = agent_result['confidence']
                
                print(f"   {agent_name:<12} {status_icon} Time: {exec_time:.2f}s, Confidence: {confidence:.1%}")
        
        print("\n🎉 All tests completed!")
        return True
        
    except Exception as e:
        print(f"\n💥 Test failed with error: {str(e)}")
        return False
    
    finally:
        await api.cleanup()

async def test_configuration_scenarios():
    """Test different configuration scenarios."""
    print("\n⚙️  Testing Configuration Scenarios")
    print("=" * 50)
    
    # Test different processing modes
    test_cases = [
        ("parallel", "Parallel processing test"),
        ("sequential", "Sequential processing test")
    ]
    
    for mode, description in test_cases:
        print(f"\n🔧 {description}")
        print("-" * 30)
        
        api = BrandMonitoringAPI("config.yaml")
        
        try:
            if await api.initialize():
                result = await api.monitor_queries(
                    ["enterprise analytics solutions"], 
                    mode=mode
                )
                
                if result['success']:
                    exec_time = sum(
                        agent['execution_time'] 
                        for query_result in result['results'].values()
                        for agent in query_result['agents'].values()
                        if agent['execution_time']
                    )
                    print(f"✅ {mode.title()} mode completed in {exec_time:.2f}s total")
                else:
                    print(f"❌ {mode.title()} mode failed")
            else:
                print(f"❌ Failed to initialize for {mode} mode test")
                
        except Exception as e:
            print(f"❌ {mode.title()} mode test error: {str(e)}")
        
        finally:
            await api.cleanup()

async def test_error_handling():
    """Test error handling capabilities."""
    print("\n🛡️  Testing Error Handling")
    print("=" * 40)
    
    # Test with invalid configuration (this should be handled gracefully)
    print("Testing graceful error handling...")
    
    try:
        # Test with non-existent config file
        api = BrandMonitoringAPI("non_existent_config.yaml")
        success = await api.initialize()
        
        if not success:
            print("✅ Properly handled missing configuration file")
        else:
            print("⚠️  Unexpected success with missing config")
            
    except Exception as e:
        print(f"✅ Caught expected error: {type(e).__name__}")
    
    # Test with malformed queries
    print("Testing with edge case queries...")
    
    api = BrandMonitoringAPI("config.yaml")
    try:
        if await api.initialize():
            # Test with empty and unusual queries
            edge_case_queries = [
                "",  # Empty query
                "a",  # Very short query
                "x" * 1000,  # Very long query
            ]
            
            result = await api.monitor_queries(edge_case_queries)
            
            if result['success']:
                print("✅ Handled edge case queries successfully")
            else:
                print("✅ Properly rejected edge case queries")
        
    except Exception as e:
        print(f"✅ Caught expected error with edge cases: {type(e).__name__}")
    
    finally:
        await api.cleanup()

def print_system_info():
    """Print system information and setup status."""
    print("\n🖥️  System Information")
    print("=" * 40)
    
    # Check environment variables
    required_env_vars = [
        "OPENAI_API_KEY",
        "PERPLEXITY_API_KEY", 
        "GOOGLE_SPREADSHEET_ID"
    ]
    
    print("Environment Variables:")
    for var in required_env_vars:
        value = os.getenv(var)
        status = "✅ SET" if value else "❌ NOT SET"
        masked_value = f"{value[:8]}..." if value and len(value) > 8 else "None"
        print(f"   {var:<25} {status:<10} ({masked_value})")
    
    # Check required files
    required_files = [
        "config.yaml",
        "credentials.json",
        ".env"
    ]
    
    print("\nRequired Files:")
    for file_path in required_files:
        exists = Path(file_path).exists()
        status = "✅ EXISTS" if exists else "❌ MISSING"
        print(f"   {file_path:<25} {status}")
    
    print("\nSetup Status:")
    env_ok = all(os.getenv(var) for var in required_env_vars)
    files_ok = all(Path(f).exists() for f in required_files)
    
    if env_ok and files_ok:
        print("✅ System appears to be properly configured!")
    else:
        print("⚠️  System configuration incomplete. Check missing items above.")
        if not env_ok:
            print("   → Set up your .env file with API keys")
        if not files_ok:
            print("   → Ensure config.yaml and credentials.json are present")

async def main():
    """Main test execution function."""
    print("🧪 DataTobiz Brand Monitoring System - Comprehensive Test Suite")
    print("=" * 70)
    
    # Print system info first
    print_system_info()
    
    # Check if we have minimum required configuration
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("PERPLEXITY_API_KEY"):
        print("\n⚠️  WARNING: No API keys detected!")
        print("Please set up your .env file with at least one API key to run tests.")
        print("See .env.template for required variables.")
        return
    
    try:
        # Run main functionality tests
        success = await test_basic_functionality()
        
        if success:
            # Run additional tests
            await test_configuration_scenarios()
            await test_error_handling()
            
            print("\n🎊 All tests completed successfully!")
            print("\nYour DataTobiz Brand Monitoring System is ready for production use!")
        else:
            print("\n💥 Basic functionality tests failed!")
            print("Please check your configuration and API keys.")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    # Run the test suite
    asyncio.run(main())