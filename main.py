"""
DataTobiz Brand Monitoring System - Main Application

This is the main entry point for the DataTobiz brand monitoring system.
It provides both CLI and programmatic interfaces for running brand monitoring workflows.
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.workflow.graph import create_workflow
from src.config.settings import get_settings, reload_settings
from src.utils.logger import setup_logging, get_logger
from src.storage.google_sheets import GoogleSheetsManager

# Initialize logging
setup_logging()
logger = get_logger(__name__)

class BrandMonitoringApp:
    """Main application class for the brand monitoring system."""
    
    def __init__(self, config_file: str = "config.yaml"):
        """Initialize the application."""
        self.config_file = config_file
        self.workflow = None
        
    async def initialize(self) -> bool:
        """Initialize the application and all components."""
        try:
            logger.info("Initializing DataTobiz Brand Monitoring System")
            
            # Load configuration
            settings = reload_settings(self.config_file)
            
            # Validate configuration
            errors = settings.validate_configuration()
            if errors:
                logger.error("Configuration validation failed:")
                for error in errors:
                    logger.error(f"  - {error}")
                return False
            
            # Create and initialize workflow
            self.workflow = await create_workflow(settings)
            
            logger.info("Application initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Application initialization failed: {str(e)}")
            return False
    
    async def run_monitoring(
        self, 
        queries: List[str], 
        processing_mode: str = "parallel",
        output_format: str = "summary"
    ) -> bool:
        """
        Run brand monitoring for the given queries.
        
        Args:
            queries: List of search queries
            processing_mode: "parallel" or "sequential"
            output_format: "summary", "detailed", or "json"
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.workflow:
                logger.error("Workflow not initialized")
                return False
            
            logger.info(f"Starting brand monitoring for {len(queries)} queries")
            
            # Execute workflow
            final_state = await self.workflow.execute_workflow(queries, processing_mode)
            
            # Output results
            self._output_results(final_state, output_format)
            
            return True
            
        except Exception as e:
            logger.error(f"Monitoring execution failed: {str(e)}")
            return False
    
    def _output_results(self, workflow_state, output_format: str):
        """Output results in the specified format."""
        if output_format == "json":
            self._output_json(workflow_state)
        elif output_format == "detailed":
            self._output_detailed(workflow_state)
        else:
            self._output_summary(workflow_state)
    
    def _output_summary(self, workflow_state):
        """Output a summary of results."""
        print("\n" + "="*60)
        print("DATATOBIZ BRAND MONITORING RESULTS")
        print("="*60)
        
        summary = workflow_state.get_progress_summary()
        
        print(f"Total Queries Processed: {summary['total_queries']}")
        print(f"Successful Executions: {summary['processed']}")
        print(f"Brand Mentions Found: {summary['brand_mentions_found']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        
        print("\nQuery Results:")
        print("-" * 40)
        
        for query, query_state in workflow_state.query_states.items():
            status = "✓ FOUND" if query_state.overall_found else "✗ NOT FOUND"
            confidence = f"({query_state.consensus_confidence:.1%})" if query_state.overall_found else ""
            
            print(f"{query[:50]:<50} {status} {confidence}")
        
        # Agent performance
        if hasattr(self.workflow, 'get_agent_performance_stats'):
            print("\nAgent Performance:")
            print("-" * 40)
            stats = self.workflow.get_agent_performance_stats()
            for agent_name, agent_stats in stats.items():
                success_rate = agent_stats['success_rate']
                avg_time = agent_stats['average_execution_time']
                print(f"{agent_name:<15} Success: {success_rate:.1%}, Avg Time: {avg_time:.2f}s")
        
        print("\n" + "="*60)
    
    def _output_detailed(self, workflow_state):
        """Output detailed results."""
        print("\n" + "="*80)
        print("DETAILED BRAND MONITORING RESULTS")
        print("="*80)
        
        for query, query_state in workflow_state.query_states.items():
            print(f"\nQuery: {query}")
            print("-" * len(query))
            print(f"Overall Found: {query_state.overall_found}")
            print(f"Consensus Confidence: {query_state.consensus_confidence:.3f}")
            
            if query_state.best_ranking:
                print(f"Best Ranking: #{query_state.best_ranking}")
            
            print("\nAgent Results:")
            for agent_name, result in query_state.agent_results.items():
                print(f"  {agent_name}:")
                print(f"    Status: {result.status}")
                if result.brand_detection:
                    print(f"    Found: {result.brand_detection.found}")
                    print(f"    Confidence: {result.brand_detection.confidence:.3f}")
                    if result.brand_detection.matches:
                        print(f"    Matches: {', '.join(result.brand_detection.matches)}")
                if result.error_message:
                    print(f"    Error: {result.error_message}")
                print(f"    Execution Time: {result.execution_time:.3f}s")
            print()
        
        print("="*80)
    
    def _output_json(self, workflow_state):
        """Output results in JSON format."""
        # Convert to JSON-serializable format
        result_data = {
            'workflow_id': workflow_state.workflow_id,
            'summary': workflow_state.get_progress_summary(),
            'queries': {}
        }
        
        for query, query_state in workflow_state.query_states.items():
            result_data['queries'][query] = {
                'found': query_state.overall_found,
                'confidence': query_state.consensus_confidence,
                'ranking': query_state.best_ranking,
                'agents': {}
            }
            
            for agent_name, result in query_state.agent_results.items():
                agent_data = {
                    'status': result.status,
                    'execution_time': result.execution_time,
                    'found': False,
                    'confidence': 0.0
                }
                
                if result.brand_detection:
                    agent_data.update({
                        'found': result.brand_detection.found,
                        'confidence': result.brand_detection.confidence,
                        'matches': result.brand_detection.matches
                    })
                
                if result.error_message:
                    agent_data['error'] = result.error_message
                
                result_data['queries'][query]['agents'][agent_name] = agent_data
        
        print(json.dumps(result_data, indent=2, default=str))
    
    async def test_connections(self) -> bool:
        """Test connections to all services."""
        try:
            logger.info("Testing connections to all services...")
            
            if not self.workflow:
                logger.error("Workflow not initialized")
                return False
            
            # Test agent connections
            print("Testing Agent Connections:")
            print("-" * 30)
            
            all_healthy = True
            for agent_name, agent in self.workflow.agents.items():
                try:
                    healthy = await agent.test_connection()
                    status = "✓ OK" if healthy else "✗ FAILED"
                    print(f"{agent_name:<15} {status}")
                    all_healthy &= healthy
                except Exception as e:
                    print(f"{agent_name:<15} ✗ ERROR: {str(e)}")
                    all_healthy = False
            
            # Test Google Sheets connection
            print(f"\nTesting Google Sheets:")
            print("-" * 30)
            try:
                if self.workflow.storage_manager:
                    # Try to get summary stats (reads from sheet)
                    await self.workflow.storage_manager.get_summary_stats()
                    print("Google Sheets      ✓ OK")
                else:
                    print("Google Sheets      ✗ NOT INITIALIZED")
                    all_healthy = False
            except Exception as e:
                print(f"Google Sheets      ✗ ERROR: {str(e)}")
                all_healthy = False
            
            print(f"\nOverall Status: {'✓ ALL SYSTEMS HEALTHY' if all_healthy else '✗ SOME SYSTEMS FAILED'}")
            return all_healthy
            
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    async def get_historical_stats(self, days_back: int = 30):
        """Get and display historical statistics."""
        try:
            if not self.workflow or not self.workflow.storage_manager:
                logger.error("Storage manager not available")
                return
            
            print(f"\nHistorical Statistics (Last {days_back} days):")
            print("="*50)
            
            # Get summary stats
            stats = await self.workflow.storage_manager.get_summary_stats()
            
            if stats:
                print(f"Total Unique Queries: {stats.get('total_unique_queries', 0)}")
                print(f"Total Results: {stats.get('total_results', 0)}")
                print(f"Brand Mentions Found: {stats.get('brand_mentions_found', 0)}")
                print(f"Detection Rate: {stats.get('detection_rate', 0):.1%}")
                print(f"Models Used: {', '.join(stats.get('models_used', []))}")
                print(f"Last Updated: {stats.get('last_updated', 'Unknown')}")
            else:
                print("No historical data available.")
            
        except Exception as e:
            logger.error(f"Failed to get historical stats: {str(e)}")
    
    async def cleanup(self):
        """Clean up application resources."""
        if self.workflow:
            await self.workflow.cleanup()

# CLI Interface
async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="DataTobiz Brand Monitoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --query "best data analytics companies" "top BI tools"
  python main.py --test-connections
  python main.py --stats --days-back 7
  python main.py --config custom_config.yaml --mode sequential
        """
    )
    
    parser.add_argument(
        "--query", "-q",
        nargs="+",
        help="Search queries to monitor (can specify multiple)"
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["parallel", "sequential"],
        default="parallel",
        help="Processing mode for agents (default: parallel)"
    )
    
    parser.add_argument(
        "--output", "-o",
        choices=["summary", "detailed", "json"],
        default="summary",
        help="Output format (default: summary)"
    )
    
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Configuration file path (default: config.yaml)"
    )
    
    parser.add_argument(
        "--test-connections", "-t",
        action="store_true",
        help="Test connections to all services"
    )
    
    parser.add_argument(
        "--stats", "-s",
        action="store_true",
        help="Show historical statistics"
    )
    
    parser.add_argument(
        "--days-back",
        type=int,
        default=30,
        help="Days back for historical stats (default: 30)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        setup_logging(log_level="DEBUG")
    
    # Create and initialize app
    app = BrandMonitoringApp(args.config)
    
    try:
        if not await app.initialize():
            print("❌ Failed to initialize application. Check logs for details.")
            return 1
        
        # Handle different commands
        if args.test_connections:
            success = await app.test_connections()
            return 0 if success else 1
        
        elif args.stats:
            await app.get_historical_stats(args.days_back)
            return 0
        
        elif args.query:
            success = await app.run_monitoring(
                queries=args.query,
                processing_mode=args.mode,
                output_format=args.output
            )
            return 0 if success else 1
        
        else:
            # Interactive mode with sample queries
            print("No queries specified. Running with sample queries...")
            sample_queries = [
                "best data analytics companies 2024",
                "top business intelligence tools",
                "leading data visualization software"
            ]
            
            success = await app.run_monitoring(
                queries=sample_queries,
                processing_mode=args.mode,
                output_format=args.output
            )
            return 0 if success else 1
    
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        return 1
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"❌ Application error: {str(e)}")
        return 1
    
    finally:
        await app.cleanup()

# Programmatic Interface
class BrandMonitoringAPI:
    """Programmatic API for the brand monitoring system."""
    
    def __init__(self, config_file: str = "config.yaml"):
        self.app = BrandMonitoringApp(config_file)
    
    async def initialize(self) -> bool:
        """Initialize the API."""
        return await self.app.initialize()
    
    async def monitor_queries(
        self, 
        queries: List[str], 
        mode: str = "parallel"
    ) -> dict:
        """
        Monitor brand mentions for given queries.
        
        Args:
            queries: List of search queries
            mode: Processing mode ("parallel" or "sequential")
            
        Returns:
            Dictionary with results
        """
        try:
            final_state = await self.app.workflow.execute_workflow(queries, mode)
            
            # Convert to simple dictionary format
            results = {}
            for query, query_state in final_state.query_states.items():
                results[query] = {
                    'found': query_state.overall_found,
                    'confidence': query_state.consensus_confidence,
                    'ranking': query_state.best_ranking,
                    'agents': {
                        agent_name: {
                            'found': result.brand_detection.found if result.brand_detection else False,
                            'confidence': result.brand_detection.confidence if result.brand_detection else 0.0,
                            'execution_time': result.execution_time,
                            'status': result.status
                        }
                        for agent_name, result in query_state.agent_results.items()
                    }
                }
            
            return {
                'success': True,
                'workflow_id': final_state.workflow_id,
                'summary': final_state.get_progress_summary(),
                'results': results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def test_connections(self) -> dict:
        """Test all service connections."""
        try:
            success = await self.app.test_connections()
            return {'success': success}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def get_stats(self, days_back: int = 30) -> dict:
        """Get historical statistics."""
        try:
            if self.app.workflow and self.app.workflow.storage_manager:
                stats = await self.app.workflow.storage_manager.get_summary_stats()
                return {'success': True, 'stats': stats}
            else:
                return {'success': False, 'error': 'Storage manager not available'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.app.cleanup()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)