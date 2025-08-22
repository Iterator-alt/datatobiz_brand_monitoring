"""
Base Agent Implementation

This module provides the abstract base class for all LLM agents in the system,
ensuring consistent interface and behavior across different providers.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from datetime import datetime

from src.workflow.state import AgentResult, AgentStatus, BrandDetectionResult
from src.utils.brand_detector import BrandDetector
from src.config.settings import LLMConfig, get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

class BaseAgent(ABC):
    """
    Abstract base class for all LLM agents.
    
    This class defines the interface that all agents must implement and provides
    common functionality for error handling, retries, and result processing.
    """
    
    def __init__(self, name: str, config: LLMConfig):
        """
        Initialize the base agent.
        
        Args:
            name: Unique name for this agent
            config: Configuration object for the LLM
        """
        self.name = name
        self.config = config
        self.brand_detector = BrandDetector()
        self.settings = get_settings()
        
        # Execution tracking
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.total_execution_time = 0.0
    
    @abstractmethod
    async def _make_llm_request(self, query: str) -> str:
        """
        Make the actual LLM request. Must be implemented by subclasses.
        
        Args:
            query: The search query to process
            
        Returns:
            Raw response from the LLM
            
        Raises:
            Exception: If the request fails
        """
        pass
    
    @abstractmethod
    def _get_model_name(self) -> str:
        """Get the model name for this agent."""
        pass
    
    def _create_search_prompt(self, query: str) -> str:
        """
        Create a standardized prompt for brand search.
        
        Args:
            query: The user's search query
            
        Returns:
            Formatted prompt for the LLM
        """
        return f"""
Please search for and provide information about: "{query}"

Focus on providing comprehensive results that include company names, tools, and platforms related to this query. Please include any mentions of data analytics companies, business intelligence tools, or related software platforms.

Include specific company names and detailed descriptions of their offerings. I'm particularly interested in understanding the competitive landscape and key players in this space.

Please provide a detailed response with specific company names and their capabilities.
"""
    
    async def execute(self, query: str, max_retries: Optional[int] = None) -> AgentResult:
        """
        Execute the agent for a given query with retry logic.
        
        Args:
            query: The search query to process
            max_retries: Maximum number of retries (uses config default if None)
            
        Returns:
            AgentResult with execution details and brand detection results
        """
        max_retries = max_retries or self.settings.workflow.max_retries
        result = AgentResult(
            agent_name=self.name,
            model_name=self._get_model_name(),
            status=AgentStatus.PENDING
        )
        
        start_time = time.time()
        self.total_executions += 1
        
        for attempt in range(max_retries + 1):
            try:
                result.retry_count = attempt
                result.status = AgentStatus.RUNNING
                
                logger.info(f"Agent {self.name} executing query: {query} (attempt {attempt + 1})")
                
                # Make the LLM request
                raw_response = await self._make_llm_request(query)
                result.raw_response = raw_response
                
                # Perform brand detection
                brand_detection = self.brand_detector.detect_brand(
                    raw_response, 
                    include_ranking=self.settings.enable_ranking_detection
                )
                result.brand_detection = brand_detection
                
                # Calculate execution time
                execution_time = time.time() - start_time
                result.execution_time = execution_time
                result.timestamp = datetime.now()
                result.status = AgentStatus.COMPLETED
                
                # Update statistics
                self.successful_executions += 1
                self.total_execution_time += execution_time
                
                logger.info(
                    f"Agent {self.name} completed successfully. "
                    f"Brand found: {brand_detection.found}, "
                    f"Confidence: {brand_detection.confidence:.2f}"
                )
                
                return result
                
            except Exception as e:
                error_msg = f"Agent {self.name} failed on attempt {attempt + 1}: {str(e)}"
                logger.error(error_msg)
                
                result.error_message = error_msg
                
                if attempt < max_retries:
                    # Wait before retry
                    await asyncio.sleep(self.settings.workflow.retry_delay * (attempt + 1))
                    continue
                else:
                    # Final failure
                    result.status = AgentStatus.FAILED
                    result.execution_time = time.time() - start_time
                    self.failed_executions += 1
                    
                    logger.error(f"Agent {self.name} failed after {max_retries + 1} attempts")
                    return result
        
        return result
    
    def _extract_token_usage(self, response_data: Any) -> Optional[Dict[str, int]]:
        """
        Extract token usage information from response data.
        Override in subclasses if the LLM provides usage data.
        
        Args:
            response_data: Raw response data from the LLM
            
        Returns:
            Dictionary with token usage information or None
        """
        return None
    
    def _estimate_cost(self, token_usage: Optional[Dict[str, int]]) -> Optional[float]:
        """
        Estimate cost based on token usage.
        Override in subclasses with provider-specific pricing.
        
        Args:
            token_usage: Token usage dictionary
            
        Returns:
            Estimated cost in USD or None
        """
        return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this agent."""
        success_rate = (
            self.successful_executions / max(self.total_executions, 1)
        )
        avg_execution_time = (
            self.total_execution_time / max(self.successful_executions, 1)
        )
        
        return {
            'agent_name': self.name,
            'total_executions': self.total_executions,
            'successful_executions': self.successful_executions,
            'failed_executions': self.failed_executions,
            'success_rate': success_rate,
            'average_execution_time': avg_execution_time,
            'total_execution_time': self.total_execution_time
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.total_execution_time = 0.0
    
    async def health_check(self) -> bool:
        """
        Perform a health check to ensure the agent is functioning.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            test_query = "test query"
            test_prompt = self._create_search_prompt(test_query)
            
            # Try a simple request with short timeout
            response = await asyncio.wait_for(
                self._make_llm_request(test_query),
                timeout=10.0
            )
            
            return bool(response and len(response.strip()) > 0)
            
        except Exception as e:
            logger.error(f"Health check failed for agent {self.name}: {str(e)}")
            return False
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, model={self._get_model_name()})"
    
    def __repr__(self) -> str:
        return self.__str__()

class AgentFactory:
    """Factory class for creating agent instances."""
    
    _agent_classes = {}
    
    @classmethod
    def register_agent(cls, agent_type: str, agent_class: type):
        """Register an agent class with the factory."""
        cls._agent_classes[agent_type] = agent_class
    
    @classmethod
    def create_agent(cls, agent_type: str, name: str, config: LLMConfig) -> BaseAgent:
        """
        Create an agent instance.
        
        Args:
            agent_type: Type of agent to create
            name: Name for the agent instance
            config: Configuration for the agent
            
        Returns:
            Agent instance
            
        Raises:
            ValueError: If agent type is not registered
        """
        if agent_type not in cls._agent_classes:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        agent_class = cls._agent_classes[agent_type]
        return agent_class(name, config)
    
    @classmethod
    def get_available_agent_types(cls) -> list:
        """Get list of available agent types."""
        return list(cls._agent_classes.keys())

# Decorator for registering agents
def register_agent(agent_type: str):
    """Decorator to register an agent class with the factory."""
    def decorator(agent_class):
        AgentFactory.register_agent(agent_type, agent_class)
        return agent_class
    return decorator