"""
Test cases for agent implementations.

This module contains comprehensive tests for all agent types to ensure
proper functionality and error handling.
"""

import pytest
import asyncio
import os
import sys
from unittest.mock import Mock, patch, AsyncMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.agents.base_agent import BaseAgent, AgentFactory
from src.agents.openai_agent import OpenAIAgent
from src.agents.perplexity_agent import PerplexityAgent
from src.config.settings import LLMConfig
from src.workflow.state import AgentStatus

class TestAgentBase:
    """Base class for agent tests with common utilities."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock LLM configuration."""
        return LLMConfig(
            name="test",
            api_key="test_key",
            model="test_model",
            max_tokens=100,
            temperature=0.1,
            timeout=10
        )
    
    def create_mock_brand_detector(self):
        """Create a mock brand detector."""
        mock_detector = Mock()
        mock_detector.detect_brand.return_value = Mock(
            found=True,
            confidence=0.8,
            matches=["DataTobiz"],
            context="Test context"
        )
        return mock_detector

class TestOpenAIAgent(TestAgentBase):
    """Test cases for OpenAI agent."""
    
    @pytest.fixture
    def agent(self, mock_config):
        """Create an OpenAI agent for testing."""
        agent = OpenAIAgent("test_openai", mock_config)
        agent.brand_detector = self.create_mock_brand_detector()
        return agent
    
    def test_agent_initialization(self, mock_config):
        """Test agent initialization."""
        agent = OpenAIAgent("test", mock_config)
        
        assert agent.name == "test"
        assert agent.config == mock_config
        assert agent.model == "test_model"
        assert agent.max_tokens == 100
        assert agent.temperature == 0.1
    
    def test_get_model_name(self, agent):
        """Test model name retrieval."""
        assert agent._get_model_name() == "test_model"
    
    def test_create_search_prompt(self, agent):
        """Test search prompt creation."""
        query = "test query"
        prompt = agent._create_search_prompt(query)
        
        assert query in prompt
        assert "comprehensive" in prompt.lower()
        assert "company names" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_execute_success(self, agent):
        """Test successful agent execution."""
        # Mock the LLM request
        mock_response = "DataTobiz is a leading data analytics company..."
        
        with patch.object(agent, '_make_llm_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await agent.execute("test query")
            
            assert result.status == AgentStatus.COMPLETED
            assert result.raw_response == mock_response
            assert result.brand_detection.found == True
            assert result.execution_time > 0
            assert result.error_message is None
    
    @pytest.mark.asyncio
    async def test_execute_failure_with_retry(self, agent):
        """Test agent execution with failure and retry."""
        with patch.object(agent, '_make_llm_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = Exception("API Error")
            
            result = await agent.execute("test query", max_retries=2)
            
            assert result.status == AgentStatus.FAILED
            assert result.error_message is not None
            assert "API Error" in result.error_message
            assert result.retry_count == 2
    
    @pytest.mark.asyncio
    async def test_health_check(self, agent):
        """Test agent health check."""
        with patch.object(agent, '_make_llm_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = "test response"
            
            is_healthy = await agent.health_check()
            assert is_healthy == True
    
    def test_performance_stats(self, agent):
        """Test performance statistics tracking."""
        # Simulate some executions
        agent.total_executions = 10
        agent.successful_executions = 8
        agent.failed_executions = 2
        agent.total_execution_time = 25.0
        
        stats = agent.get_performance_stats()
        
        assert stats['total_executions'] == 10
        assert stats['successful_executions'] == 8
        assert stats['failed_executions'] == 2
        assert stats['success_rate'] == 0.8
        assert stats['average_execution_time'] == 25.0 / 8

class TestPerplexityAgent(TestAgentBase):
    """Test cases for Perplexity agent."""
    
    @pytest.fixture
    def agent(self, mock_config):
        """Create a Perplexity agent for testing."""
        agent = PerplexityAgent("test_perplexity", mock_config)
        agent.brand_detector = self.create_mock_brand_detector()
        return agent
    
    def test_agent_initialization(self, mock_config):
        """Test agent initialization."""
        agent = PerplexityAgent("test", mock_config)
        
        assert agent.name == "test"
        assert agent.config == mock_config
        assert agent.model == "test_model"
        assert "api.perplexity.ai" in agent.api_base
    
    @pytest.mark.asyncio
    async def test_session_management(self, agent):
        """Test HTTP session management."""
        # Test session creation
        session = await agent._get_session()
        assert session is not None
        
        # Test session reuse
        session2 = await agent._get_session()
        assert session is session2
        
        # Test session cleanup
        await agent._close_session()
    
    @pytest.mark.asyncio
    async def test_execute_with_mock_response(self, agent):
        """Test execution with mocked HTTP response."""
        mock_response_data = {
            "choices": [{
                "message": {
                    "content": "DataTobiz is mentioned in recent analytics reports..."
                }
            }],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "total_tokens": 150
            }
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Mock the context manager
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_response_data
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await agent.execute("test query")
            
            assert result.status == AgentStatus.COMPLETED
            assert result.brand_detection.found == True
    
    def test_get_model_info(self, agent):
        """Test model information retrieval."""
        info = agent.get_model_info()
        
        assert info['provider'] == 'Perplexity'
        assert info['model'] == 'test_model'
        assert 'web_search' in info['features']
        assert 'real_time_data' in info['features']

class TestAgentFactory:
    """Test cases for agent factory."""
    
    def test_register_and_create_agent(self):
        """Test agent registration and creation."""
        # Create mock agent class
        class MockAgent(BaseAgent):
            async def _make_llm_request(self, query: str) -> str:
                return "mock response"
            
            def _get_model_name(self) -> str:
                return "mock_model"
        
        # Register the agent
        AgentFactory.register_agent("mock", MockAgent)
        
        # Test creation
        config = LLMConfig(name="test", api_key="test", model="test")
        agent = AgentFactory.create_agent("mock", "test_agent", config)
        
        assert isinstance(agent, MockAgent)
        assert agent.name == "test_agent"
    
    def test_unknown_agent_type(self):
        """Test creation of unknown agent type."""
        config = LLMConfig(name="test", api_key="test", model="test")
        
        with pytest.raises(ValueError, match="Unknown agent type"):
            AgentFactory.create_agent("unknown", "test", config)
    
    def test_get_available_agent_types(self):
        """Test getting available agent types."""
        types = AgentFactory.get_available_agent_types()
        
        # Should include at least openai and perplexity
        assert "openai" in types
        assert "perplexity" in types

class TestAgentIntegration:
    """Integration tests for agent functionality."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_openai_agent(self):
        """Test real OpenAI agent (requires API key)."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")
        
        config = LLMConfig(
            name="openai",
            api_key=api_key,
            model="gpt-3.5-turbo",
            max_tokens=100,
            temperature=0.1
        )
        
        agent = OpenAIAgent("test", config)
        
        # Test health check
        healthy = await agent.health_check()
        assert healthy == True
        
        # Test execution with simple query
        result = await agent.execute("List top 3 data analytics companies")
        
        assert result.status == AgentStatus.COMPLETED
        assert result.raw_response is not None
        assert len(result.raw_response) > 0
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_perplexity_agent(self):
        """Test real Perplexity agent (requires API key)."""
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            pytest.skip("PERPLEXITY_API_KEY not set")
        
        config = LLMConfig(
            name="perplexity",
            api_key=api_key,
            model="llama-3.1-sonar-small-128k-online",
            max_tokens=100,
            temperature=0.1
        )
        
        agent = PerplexityAgent("test", config)
        
        try:
            # Test health check
            healthy = await agent.test_connection()
            assert healthy == True
            
            # Test execution
            result = await agent.execute("Current top data analytics companies")
            
            assert result.status == AgentStatus.COMPLETED
            assert result.raw_response is not None
            
        finally:
            await agent._close_session()

# Test configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])