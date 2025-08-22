"""
Test cases for workflow components.

This module contains tests for the LangGraph workflow implementation,
state management, and overall system integration.
"""

import pytest
import asyncio
import os
import sys
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.workflow.state import (
    WorkflowState, QueryState, AgentResult, BrandDetectionResult, AgentStatus,
    state_to_dict, dict_to_state, update_agent_result, finalize_query_state
)
from src.workflow.graph import BrandMonitoringWorkflow, create_workflow
from src.config.settings import LLMConfig, GoogleSheetsConfig, BrandConfig, WorkflowConfig

class TestWorkflowState:
    """Test cases for workflow state management."""
    
    @pytest.fixture
    def sample_workflow_state(self):
        """Create a sample workflow state for testing."""
        return WorkflowState(
            queries=["test query 1", "test query 2"],
            target_agents=["openai", "perplexity"],
            workflow_id="test-workflow-123",
            processing_mode="parallel"
        )
    
    @pytest.fixture
    def sample_query_state(self):
        """Create a sample query state for testing."""
        return QueryState(
            query="test query",
            query_id="test-query-123"
        )
    
    @pytest.fixture
    def sample_agent_result(self):
        """Create a sample agent result for testing."""
        brand_detection = BrandDetectionResult(
            found=True,
            confidence=0.85,
            matches=["DataTobiz"],
            context="DataTobiz is a leading analytics company"
        )
        
        return AgentResult(
            agent_name="test_agent",
            model_name="test_model",
            status=AgentStatus.COMPLETED,
            raw_response="Test response containing DataTobiz mention",
            brand_detection=brand_detection,
            execution_time=2.5
        )
    
    def test_workflow_state_initialization(self, sample_workflow_state):
        """Test workflow state initialization."""
        state = sample_workflow_state
        
        assert len(state.queries) == 2
        assert state.workflow_id == "test-workflow-123"
        assert state.processing_mode == "parallel"
        assert state.current_query_index == 0
        assert len(state.query_states) == 0
    
    def test_add_query_state(self, sample_workflow_state, sample_query_state):
        """Test adding query state to workflow."""
        workflow_state = sample_workflow_state
        query_state = sample_query_state
        
        workflow_state.add_query_state("test query", query_state)
        
        assert len(workflow_state.query_states) == 1
        assert workflow_state.get_query_state("test query") == query_state
    
    def test_mark_query_completed(self, sample_workflow_state):
        """Test marking query as completed."""
        state = sample_workflow_state
        
        state.mark_query_completed("test query 1")
        
        assert "test query 1" in state.completed_queries
        assert state.successful_queries == 1
    
    def test_mark_query_failed(self, sample_workflow_state):
        """Test marking query as failed."""
        state = sample_workflow_state
        
        state.mark_query_failed("test query 1", "Test error")
        
        assert "test query 1" in state.failed_queries
        assert state.failed_queries_count == 1
    
    def test_update_summary_stats(self, sample_workflow_state):
        """Test updating summary statistics."""
        state = sample_workflow_state
        
        # Add some completed and failed queries
        state.mark_query_completed("test query 1")
        state.mark_query_failed("test query 2", "Test error")
        
        state.update_summary_stats()
        
        assert state.total_queries == 2
        assert state.successful_queries == 1
        assert state.failed_queries_count == 1
    
    def test_get_progress_summary(self, sample_workflow_state):
        """Test getting progress summary."""
        state = sample_workflow_state
        
        # Add some progress
        state.mark_query_completed("test query 1")
        state.update_summary_stats()
        
        summary = state.get_progress_summary()
        
        assert summary['total_queries'] == 2
        assert summary['processed'] == 1
        assert summary['remaining'] == 1
        assert summary['success_rate'] == 1.0
    
    def test_is_complete(self, sample_workflow_state):
        """Test workflow completion check."""
        state = sample_workflow_state
        
        # Initially not complete
        assert not state.is_complete()
        
        # Complete all queries
        state.mark_query_completed("test query 1")
        state.mark_query_completed("test query 2")
        
        assert state.is_complete()

class TestStateUtilities:
    """Test state utility functions."""
    
    @pytest.fixture
    def sample_workflow_state(self):
        """Create a sample workflow state."""
        return WorkflowState(
            queries=["test query"],
            workflow_id="test-123"
        )
    
    def test_state_to_dict_conversion(self, sample_workflow_state):
        """Test converting state to dictionary."""
        state = sample_workflow_state
        state_dict = state_to_dict(state)
        
        assert isinstance(state_dict, dict)
        assert state_dict['workflow_id'] == "test-123"
        assert state_dict['queries'] == ["test query"]
    
    def test_dict_to_state_conversion(self, sample_workflow_state):
        """Test converting dictionary back to state."""
        original_state = sample_workflow_state
        state_dict = state_to_dict(original_state)
        restored_state = dict_to_state(state_dict)
        
        assert restored_state.workflow_id == original_state.workflow_id
        assert restored_state.queries == original_state.queries
    
    def test_update_agent_result(self, sample_workflow_state):
        """Test updating agent result in workflow state."""
        state = sample_workflow_state
        query = "test query"
        
        # Add query state first
        query_state = QueryState(query=query)
        state.add_query_state(query, query_state)
        
        # Create agent result
        brand_detection = BrandDetectionResult(found=True, confidence=0.8)
        agent_result = AgentResult(
            agent_name="test_agent",
            model_name="test_model",
            brand_detection=brand_detection
        )
        
        # Update agent result
        updated_state = update_agent_result(state, query, "test_agent", agent_result)
        
        query_state = updated_state.get_query_state(query)
        assert "test_agent" in query_state.agent_results
        assert query_state.overall_found == True
    
    def test_finalize_query_state(self, sample_workflow_state):
        """Test finalizing query state."""
        state = sample_workflow_state
        query = "test query"
        
        # Add query state
        query_state = QueryState(query=query)
        state.add_query_state(query, query_state)
        
        # Finalize query state
        finalized_state = finalize_query_state(state, query)
        
        final_query_state = finalized_state.get_query_state(query)
        assert final_query_state.status == AgentStatus.COMPLETED
        assert final_query_state.end_time is not None
        assert final_query_state.total_execution_time is not None

class TestBrandMonitoringWorkflow:
    """Test cases for the main workflow class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        llm_configs = {
            "openai": LLMConfig(name="openai", api_key="test_key", model="gpt-4"),
            "perplexity": LLMConfig(name="perplexity", api_key="test_key", model="llama-3.1-sonar-small-128k-online")
        }
        
        mock_config = Mock()
        mock_config.llm_configs = llm_configs
        mock_config.google_sheets = GoogleSheetsConfig(spreadsheet_id="test_id")
        mock_config.brand = BrandConfig()
        mock_config.workflow = WorkflowConfig()
        mock_config.enable_ranking_detection = False
        
        return mock_config
    
    @pytest.fixture
    def workflow(self, mock_config):
        """Create a workflow instance for testing."""
        return BrandMonitoringWorkflow(mock_config)
    
    def test_workflow_initialization(self, workflow):
        """Test workflow initialization."""
        assert workflow.agents == {}
        assert workflow.storage_manager is None
        assert workflow.graph is None
        assert len(workflow.execution_history) == 0
    
    @pytest.mark.asyncio
    async def test_agent_initialization_success(self, workflow):
        """Test successful agent initialization."""
        # Mock agent classes
        mock_openai_agent = Mock()
        mock_openai_agent.health_check = AsyncMock(return_value=True)
        
        mock_perplexity_agent = Mock()
        mock_perplexity_agent.health_check = AsyncMock(return_value=True)
        
        with patch('src.workflow.graph.OpenAIAgent', return_value=mock_openai_agent):
            with patch('src.workflow.graph.PerplexityAgent', return_value=mock_perplexity_agent):
                success = await workflow._initialize_agents()
                
                assert success == True
                assert len(workflow.agents) == 2
                assert "openai" in workflow.agents
                assert "perplexity" in workflow.agents
    
    @pytest.mark.asyncio
    async def test_agent_initialization_failure(self, workflow):
        """Test agent initialization with health check failure."""
        # Mock agent with failed health check
        mock_agent = Mock()
        mock_agent.health_check = AsyncMock(return_value=False)
        
        with patch('src.workflow.graph.OpenAIAgent', return_value=mock_agent):
            with patch('src.workflow.graph.PerplexityAgent', return_value=mock_agent):
                success = await workflow._initialize_agents()
                
                assert success == False
    
    @pytest.mark.asyncio
    async def test_storage_initialization_success(self, workflow):
        """Test successful storage initialization."""
        mock_storage = Mock()
        mock_storage.initialize = AsyncMock(return_value=True)
        
        with patch('src.workflow.graph.GoogleSheetsManager', return_value=mock_storage):
            success = await workflow._initialize_storage()
            
            assert success == True
            assert workflow.storage_manager == mock_storage
    
    @pytest.mark.asyncio
    async def test_storage_initialization_failure(self, workflow):
        """Test storage initialization failure."""
        mock_storage = Mock()
        mock_storage.initialize = AsyncMock(return_value=False)
        
        with patch('src.workflow.graph.GoogleSheetsManager', return_value=mock_storage):
            success = await workflow._initialize_storage()
            
            assert success == False
    
    def test_graph_building(self, workflow):
        """Test LangGraph building."""
        workflow._build_graph()
        
        assert workflow.graph is not None
    
    @pytest.mark.asyncio
    async def test_full_workflow_initialization(self, workflow):
        """Test complete workflow initialization."""
        # Mock all dependencies
        mock_agent = Mock()
        mock_agent.health_check = AsyncMock(return_value=True)
        
        mock_storage = Mock()
        mock_storage.initialize = AsyncMock(return_value=True)
        
        with patch('src.workflow.graph.OpenAIAgent', return_value=mock_agent):
            with patch('src.workflow.graph.PerplexityAgent', return_value=mock_agent):
                with patch('src.workflow.graph.GoogleSheetsManager', return_value=mock_storage):
                    success = await workflow.initialize()
                    
                    assert success == True
                    assert len(workflow.agents) == 2
                    assert workflow.storage_manager is not None
                    assert workflow.graph is not None

class TestWorkflowExecution:
    """Test workflow execution scenarios."""
    
    @pytest.fixture
    def mock_workflow(self):
        """Create a mock workflow for testing execution."""
        workflow = Mock(spec=BrandMonitoringWorkflow)
        
        # Mock agents
        mock_agent = Mock()
        mock_agent.execute = AsyncMock(return_value=AgentResult(
            agent_name="test_agent",
            model_name="test_model",
            status=AgentStatus.COMPLETED,
            brand_detection=BrandDetectionResult(found=True, confidence=0.8)
        ))
        
        workflow.agents = {"test_agent": mock_agent}
        
        # Mock storage
        mock_storage = Mock()
        mock_storage.store_single_result = AsyncMock(return_value=True)
        workflow.storage_manager = mock_storage
        
        return workflow
    
    @pytest.mark.asyncio
    async def test_single_agent_execution(self, mock_workflow):
        """Test executing a single agent."""
        agent = list(mock_workflow.agents.values())[0]
        result = await agent.execute("test query")
        
        assert result.status == AgentStatus.COMPLETED
        assert result.brand_detection.found == True
    
    @pytest.mark.asyncio
    async def test_parallel_agent_execution(self):
        """Test parallel execution of multiple agents."""
        # Create mock agents
        async def mock_execute(query):
            await asyncio.sleep(0.1)  # Simulate work
            return AgentResult(
                agent_name="test_agent",
                model_name="test_model",
                status=AgentStatus.COMPLETED
            )
        
        agents = {
            "agent1": Mock(execute=mock_execute),
            "agent2": Mock(execute=mock_execute)
        }
        
        # Execute in parallel
        tasks = [agent.execute("test query") for agent in agents.values()]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 2
        assert all(result.status == AgentStatus.COMPLETED for result in results)

class TestWorkflowIntegration:
    """Integration tests for complete workflow scenarios."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_create_workflow_function(self):
        """Test the create_workflow factory function."""
        # Mock configuration
        mock_config = Mock()
        mock_config.llm_configs = {
            "test": LLMConfig(name="test", api_key="test_key", model="test_model")
        }
        mock_config.google_sheets = GoogleSheetsConfig(spreadsheet_id="test_id")
        
        # Mock workflow initialization
        with patch.object(BrandMonitoringWorkflow, 'initialize', new_callable=AsyncMock) as mock_init:
            mock_init.return_value = True
            
            workflow = await create_workflow(mock_config)
            
            assert isinstance(workflow, BrandMonitoringWorkflow)
            mock_init.assert_called_once()
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_create_workflow_initialization_failure(self):
        """Test create_workflow with initialization failure."""
        mock_config = Mock()
        
        with patch.object(BrandMonitoringWorkflow, 'initialize', new_callable=AsyncMock) as mock_init:
            mock_init.return_value = False
            
            with pytest.raises(RuntimeError, match="Failed to initialize workflow"):
                await create_workflow(mock_config)

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