"""
Perplexity Agent Implementation

This module provides the Perplexity-specific agent implementation for brand monitoring.
Perplexity provides web-search enhanced responses, making it ideal for real-time brand monitoring.
"""

import asyncio
import aiohttp
from typing import Dict, Any, Optional
import json

from src.agents.base_agent import BaseAgent, register_agent
from src.config.settings import LLMConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

@register_agent("perplexity")
class PerplexityAgent(BaseAgent):
    """
    Perplexity agent for brand monitoring with web search capabilities.
    
    This agent uses Perplexity's online search-enhanced LLM to provide
    real-time, web-informed responses about brand mentions.
    """
    
    def __init__(self, name: str, config: LLMConfig):
        """Initialize the Perplexity agent."""
        super().__init__(name, config)
        
        # Perplexity API configuration
        self.api_base = "https://api.perplexity.ai"
        self.api_key = config.api_key
        self.model = config.model or "llama-3.1-sonar-small-128k-online"
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        
        # HTTP session for efficient connection reuse
        self._session = None
        
        logger.info(f"Initialized Perplexity agent with model: {self.model}")
    
    def _get_model_name(self) -> str:
        """Get the model name for this agent."""
        return self.model
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def _close_session(self):
        """Close HTTP session if it exists."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _create_search_prompt(self, query: str) -> str:
        """
        Create a Perplexity-specific prompt for brand search.
        
        Args:
            query: The user's search query
            
        Returns:
            Formatted prompt optimized for Perplexity's web search capabilities
        """
        return f"""
Search the web and provide current information about: "{query}"

Please include:

1. Current market leaders and top-rated companies/tools in this space
2. Recent developments, new products, or industry changes
3. Comparative analysis of different solutions
4. User reviews and expert opinions from recent sources
5. Any emerging companies or innovative solutions
6. Market trends and competitive landscape updates

Focus on providing up-to-date, comprehensive information that includes specific company names, product names, and detailed descriptions. Include both established market leaders and emerging players.

Please cite sources where possible and ensure the information is current and accurate. Include specific details about each company's offerings, target markets, and key differentiators.

Search for recent articles, reviews, and industry reports to ensure the most current information.
"""
    
    async def _make_llm_request(self, query: str) -> str:
        """
        Make a request to Perplexity's API.
        
        Args:
            query: The search query to process
            
        Returns:
            Raw response from Perplexity
            
        Raises:
            Exception: If the API request fails
        """
        try:
            session = await self._get_session()
            prompt = self._create_search_prompt(query)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a research assistant with access to current web information. Provide detailed, up-to-date information about companies, tools, and market trends in data analytics and business intelligence. Always search for the most recent information available."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "search_domain_filter": ["perplexity.ai"],  # Allow web search
                "return_citations": True,
                "search_recency_filter": "month"  # Focus on recent information
            }
            
            logger.debug(f"Making Perplexity request for query: {query}")
            
            async with session.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Perplexity API error {response.status}: {error_text}")
                
                response_data = await response.json()
                
                if 'choices' not in response_data or not response_data['choices']:
                    raise Exception("No choices returned from Perplexity API")
                
                content = response_data['choices'][0]['message']['content']
                if not content:
                    raise Exception("Empty content returned from Perplexity")
                
                logger.debug(f"Perplexity response length: {len(content)} characters")
                
                # Log citation information if available
                if 'citations' in response_data:
                    logger.debug(f"Perplexity returned {len(response_data['citations'])} citations")
                
                return content.strip()
                
        except aiohttp.ClientError as e:
            raise Exception(f"Perplexity connection error: {str(e)}")
        except asyncio.TimeoutError as e:
            raise Exception(f"Perplexity request timeout: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"Perplexity response parsing error: {str(e)}")
        except Exception as e:
            if "Perplexity" not in str(e):
                raise Exception(f"Unexpected Perplexity error: {str(e)}")
            raise
    
    def _extract_token_usage(self, response_data: Any) -> Optional[Dict[str, int]]:
        """Extract token usage from Perplexity response."""
        if isinstance(response_data, dict) and 'usage' in response_data:
            usage = response_data['usage']
            return {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0)
            }
        return None
    
    def _estimate_cost(self, token_usage: Optional[Dict[str, int]]) -> Optional[float]:
        """
        Estimate cost for Perplexity API usage.
        
        Note: Prices are approximate and may change. Update regularly.
        """
        if not token_usage:
            return None
        
        # Approximate pricing for Perplexity online models (as of 2024)
        # Update these values based on current Perplexity pricing
        if "sonar" in self.model.lower():
            price_per_1k_tokens = 0.005  # $0.005 per 1K tokens for sonar models
        else:
            price_per_1k_tokens = 0.002  # $0.002 per 1K tokens for base models
        
        total_tokens = token_usage.get('total_tokens', 0)
        return (total_tokens / 1000) * price_per_1k_tokens
    
    async def test_connection(self) -> bool:
        """
        Test the connection to Perplexity API.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            session = await self._get_session()
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": "Hello, this is a connection test."}
                ],
                "max_tokens": 10,
                "temperature": 0
            }
            
            async with session.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                
                if response.status == 200:
                    response_data = await response.json()
                    return bool(
                        response_data.get('choices') and 
                        response_data['choices'][0].get('message', {}).get('content')
                    )
                else:
                    return False
                    
        except Exception as e:
            logger.error(f"Perplexity connection test failed: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        return {
            'provider': 'Perplexity',
            'model': self.model,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'timeout': self.config.timeout,
            'features': ['web_search', 'real_time_data', 'citations']
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._close_session()

# Utility function for creating Perplexity agents
def create_perplexity_agent(
    name: str = "perplexity", 
    api_key: str = None, 
    model: str = "llama-3.1-sonar-small-128k-online"
) -> PerplexityAgent:
    """
    Utility function to create a Perplexity agent with default configuration.
    
    Args:
        name: Name for the agent
        api_key: Perplexity API key (uses environment variable if None)
        model: Model to use
        
    Returns:
        Configured Perplexity agent
    """
    from src.config.settings import get_settings
    
    settings = get_settings()
    config = LLMConfig(
        name=name,
        api_key=api_key or settings.perplexity_api_key,
        model=model,
        max_tokens=1000,
        temperature=0.1,
        timeout=30
    )
    
    return PerplexityAgent(name, config)