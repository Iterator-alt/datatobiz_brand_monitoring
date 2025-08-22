"""
Integration tests for the complete DataTobiz Brand Monitoring System.

This module contains end-to-end tests that verify the entire system
works correctly with real or mocked external services.
"""

import pytest
import asyncio
import os
import sys
from unittest.mock import Mock, patch, AsyncMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import BrandMonitoringAPI, BrandMonitoringApp
from src.workflow.graph import create_workflow
from src.config.settings import get_settings, Settings
from src.workflow.state import WorkflowState, AgentStatus

class TestEndToEndWorkflow:
    """Test complete end-to-end workflow execution."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for integration tests."""
        with patch('src.config.settings.get_settings') as mock_get_settings:
            mock_settings = Mock()
            
            # Mock LLM configs
            from src.config.settings import LLMConfig
            mock_settings.llm_configs = {
                "openai": LLMConfig(name="openai", api_key="test_key", model="gpt-4"),
                "perplexity": LLMConfig(name="perplexity", api_key="test_key", model="llama-3.1-sonar-small-128k-online")
            }
            
            # Mock other configs
            from src.config.settings import GoogleSheetsConfig, BrandConfig, WorkflowConfig
            mock_settings.google_sheets = GoogleSheetsConfig(spreadsheet_id="test_id")
            mock_settings.brand = BrandConfig()
            mock_settings.workflow = WorkflowConfig()
            mock_settings.enable_ranking_detection = False
            mock_settings.openai_api_key = "test_key"
            mock_settings.perplexity_api_key = "test_key"
            
            mock_get_settings.return_value = mock_settings
            yield mock_settings
    
    @pytest.mark.asyncio
    async def test_complete_workflow_execution(self, mock_settings):
        """Test complete workflow from start to finish."""
        
        # Mock agent responses
        mock_openai_response = "OpenAI found DataTobiz as a leading analytics company with excellent tools."
        mock_perplexity_response = "Perplexity search shows DataTobiz among top data analytics platforms."
        
        # Mock brand detection results
        from src.workflow.state import BrandDetectionResult
        mock_brand_detection = BrandDetectionResult(
            found=True,
            confidence=0.85,
            matches=["DataTobiz"],
            context="Leading analytics company"
        )
        
        with patch('src.agents.openai_agent.OpenAIAgent') as mock_openai_class:
            with patch('src.agents.perplexity_agent.PerplexityAgent') as mock_perplexity_class:
                with patch('src.storage.google_sheets.GoogleSheetsManager') as mock_storage_class:
                    
                    # Setup OpenAI agent mock
                    mock_openai_agent = Mock()
                    mock_openai_agent.health_check = AsyncMock(return_value=True)
                    mock_openai_agent.execute = AsyncMock(return_value=Mock(
                        agent_name="openai",
                        model_name="gpt-4",
                        status=AgentStatus.COMPLETED,
                        raw_response=mock_openai_response,
                        brand_detection=mock_brand_detection,
                        execution_time=2.5,
                        error_message=None,
                        retry_count=0
                    ))
                    mock_openai_class.return_value = mock_openai_agent
                    
                    # Setup Perplexity agent mock
                    mock_perplexity_agent = Mock()
                    mock_perplexity_agent.health_check = AsyncMock(return_value=True)
                    mock_perplexity_agent.execute = AsyncMock(return_value=Mock(
                        agent_name="perplexity",
                        model_name="llama-3.1-sonar-small-128k-online",
                        status=AgentStatus.COMPLETED,
                        raw_response=mock_perplexity_response,
                        brand_detection=mock_brand_detection,
                        execution_time=3.2,
                        error_message=None,
                        retry_count=0
                    ))
                    mock_perplexity_class.return_value = mock_perplexity_agent
                    
                    # Setup storage mock
                    mock_storage = Mock()
                    mock_storage.initialize = AsyncMock(return_value=True)
                    mock_storage.store_single_result = AsyncMock(return_value=True)
                    mock_storage.cleanup = AsyncMock()
                    mock_storage_class.return_value = mock_storage
                    
                    # Create and execute workflow
                    workflow = await create_workflow()
                    
                    test_queries = [
                        "best data analytics companies",
                        "top business intelligence tools"
                    ]
                    
                    result = await workflow.execute_workflow(test_queries, "parallel")
                    
                    # Verify results
                    assert isinstance(result, WorkflowState)
                    assert len(result.query_states) == 2
                    assert result.total_queries == 2
                    assert result.successful_queries == 2
                    
                    # Check individual query results
                    for query in test_queries:
                        query_state = result.get_query_state(query)
                        assert query_state is not None
                        assert query_state.overall_found == True
                        assert len(query_state.agent_results) == 2
                        assert "openai" in query_state.agent_results
                        assert "perplexity" in query_state.agent_results
                    
                    # Verify agent execution
                    assert mock_openai_agent.execute.call_count == 2
                    assert mock_perplexity_agent.execute.call_count == 2
                    
                    # Verify storage operations
                    assert mock_storage.store_single_result.call_count == 4  # 2 queries × 2 agents
                    
                    # Cleanup
                    await workflow.cleanup()

class TestBrandMonitoringAPI:
    """Test the programmatic API interface."""
    
    @pytest.fixture
    def mock_app(self):
        """Create a mock BrandMonitoringApp for testing."""
        app = Mock(spec=BrandMonitoringApp)
        app.initialize = AsyncMock(return_value=True)
        app.test_connections = AsyncMock(return_value=True)
        app.cleanup = AsyncMock()
        
        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.execute_workflow = AsyncMock(return_value=Mock(
            workflow_id="test-123",
            query_states={
                "test query": Mock(
                    overall_found=True,
                    consensus_confidence=0.8,
                    best_ranking=None,
                    agent_results={
                        "openai": Mock(
                            brand_detection=Mock(found=True, confidence=0.8),
                            execution_time=2.5,
                            status=AgentStatus.COMPLETED
                        )
                    }
                )
            },
            get_progress_summary=Mock(return_value={
                "total_queries": 1,
                "processed": 1,
                "brand_mentions_found": 1,
                "success_rate": 1.0
            })
        ))
        mock_workflow.storage_manager = Mock()
        mock_workflow.storage_manager.get_summary_stats = AsyncMock(return_value={
            "total_results": 10,
            "brand_mentions_found": 3,
            "detection_rate": 0.3
        })
        
        app.workflow = mock_workflow
        return app
    
    @pytest.mark.asyncio
    async def test_api_initialization(self, mock_app):
        """Test API initialization."""
        with patch('main.BrandMonitoringApp', return_value=mock_app):
            api = BrandMonitoringAPI()
            success = await api.initialize()
            
            assert success == True
            mock_app.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_api_monitor_queries(self, mock_app):
        """Test monitoring queries through API."""
        with patch('main.BrandMonitoringApp', return_value=mock_app):
            api = BrandMonitoringAPI()
            api.app = mock_app
            
            result = await api.monitor_queries(["test query"], "parallel")
            
            assert result['success'] == True
            assert 'workflow_id' in result
            assert 'summary' in result
            assert 'results' in result
            assert result['results']['test query']['found'] == True
    
    @pytest.mark.asyncio
    async def test_api_test_connections(self, mock_app):
        """Test connection testing through API."""
        with patch('main.BrandMonitoringApp', return_value=mock_app):
            api = BrandMonitoringAPI()
            api.app = mock_app
            
            result = await api.test_connections()
            
            assert result['success'] == True
            mock_app.test_connections.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_api_get_stats(self, mock_app):
        """Test getting statistics through API."""
        with patch('main.BrandMonitoringApp', return_value=mock_app):
            api = BrandMonitoringAPI()
            api.app = mock_app
            
            result = await api.get_stats()
            
            assert result['success'] == True
            assert 'stats' in result
            assert result['stats']['total_results'] == 10

class TestBrandMonitoringApp:
    """Test the main application class."""
    
    @pytest.fixture
    def mock_workflow(self):
        """Create a mock workflow for testing."""
        workflow = Mock()
        workflow.execute_workflow = AsyncMock(return_value=Mock(
            query_states={
                "test query": Mock(
                    overall_found=True,
                    consensus_confidence=0.85,
                    agent_results={
                        "openai": Mock(
                            status=AgentStatus.COMPLETED,
                            brand_detection=Mock(found=True, confidence=0.8, matches=["DataTobiz"]),
                            execution_time=2.5,
                            error_message=None
                        )
                    }
                )
            },
            get_progress_summary=Mock(return_value={
                "total_queries": 1,
                "processed": 1,
                "brand_mentions_found": 1,
                "success_rate": 1.0
            })
        ))
        workflow.agents = {
            "openai": Mock(test_connection=AsyncMock(return_value=True))
        }
        workflow.storage_manager = Mock(
            get_summary_stats=AsyncMock(return_value={"total_results": 5})
        )
        workflow.get_agent_performance_stats = Mock(return_value={
            "openai": {"success_rate": 1.0, "average_execution_time": 2.5}
        })
        workflow.cleanup = AsyncMock()
        
        return workflow
    
    @pytest.mark.asyncio
    async def test_app_initialization_success(self):
        """Test successful app initialization."""
        with patch('main.reload_settings') as mock_reload:
            with patch('main.create_workflow') as mock_create:
                
                # Mock settings
                mock_settings = Mock()
                mock_settings.validate_configuration.return_value = []
                mock_reload.return_value = mock_settings
                
                # Mock workflow creation
                mock_workflow = Mock()
                mock_create.return_value = mock_workflow
                
                app = BrandMonitoringApp()
                success = await app.initialize()
                
                assert success == True
                assert app.workflow == mock_workflow
    
    @pytest.mark.asyncio
    async def test_app_initialization_config_errors(self):
        """Test app initialization with configuration errors."""
        with patch('main.reload_settings') as mock_reload:
            
            # Mock settings with validation errors
            mock_settings = Mock()
            mock_settings.validate_configuration.return_value = ["Missing API key"]
            mock_reload.return_value = mock_settings
            
            app = BrandMonitoringApp()
            success = await app.initialize()
            
            assert success == False
    
    @pytest.mark.asyncio
    async def test_app_run_monitoring(self, mock_workflow):
        """Test running monitoring through app."""
        app = BrandMonitoringApp()
        app.workflow = mock_workflow
        
        with patch.object(app, '_output_results') as mock_output:
            success = await app.run_monitoring(["test query"], "parallel", "summary")
            
            assert success == True
            mock_workflow.execute_workflow.assert_called_once_with(["test query"], "parallel")
            mock_output.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_app_test_connections(self, mock_workflow):
        """Test connection testing through app."""
        app = BrandMonitoringApp()
        app.workflow = mock_workflow
        
        with patch('builtins.print'):  # Mock print to avoid output during tests
            success = await app.test_connections()
            
            assert success == True
    
    @pytest.mark.asyncio
    async def test_app_get_historical_stats(self, mock_workflow):
        """Test getting historical stats through app."""
        app = BrandMonitoringApp()
        app.workflow = mock_workflow
        
        with patch('builtins.print'):  # Mock print to avoid output during tests
            await app.get_historical_stats(30)
            
            mock_workflow.storage_manager.get_summary_stats.assert_called_once()

class TestRealIntegration:
    """Integration tests with real services (requires API keys)."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_real_end_to_end_workflow(self):
        """Test real end-to-end workflow with actual API calls."""
        # Skip if no API keys available
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("PERPLEXITY_API_KEY"):
            pytest.skip("No API keys available for real integration test")
        
        # Mock Google Sheets to avoid real spreadsheet operations
        with patch('src.storage.google_sheets.GoogleSheetsManager') as mock_storage_class:
            mock_storage = Mock()
            mock_storage.initialize = AsyncMock(return_value=True)
            mock_storage.store_single_result = AsyncMock(return_value=True)
            mock_storage.cleanup = AsyncMock()
            mock_storage_class.return_value = mock_storage
            
            try:
                api = BrandMonitoringAPI()
                
                # Initialize with real config
                success = await api.initialize()
                if not success:
                    pytest.skip("Failed to initialize API with real configuration")
                
                # Test with simple query
                result = await api.monitor_queries(["top analytics companies"], "sequential")
                
                assert result['success'] == True
                assert len(result['results']) == 1
                
                # Check that we got real responses
                query_result = list(result['results'].values())[0]
                assert len(query_result['agents']) > 0
                
                # At least one agent should have completed successfully
                completed_agents = [
                    agent for agent in query_result['agents'].values()
                    if agent['status'] == 'completed'
                ]
                assert len(completed_agents) > 0
                
            finally:
                await api.cleanup()

# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])