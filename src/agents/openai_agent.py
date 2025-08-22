"""
OpenAI Agent Implementation

This module provides the OpenAI-specific agent implementation for brand monitoring.
"""

import asyncio
from typing import Dict, Any, Optional
import openai
from openai import AsyncOpenAI

from src.agents.base_agent import BaseAgent, register_agent
from src.config.settings import LLMConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

@register_agent("openai")
class OpenAIAgent(BaseAgent):
    """
    OpenAI GPT agent for brand monitoring.
    
    This agent uses OpenAI's GPT models to search for and analyze brand mentions
    in response to user queries.
    """
    
    def __init__(self, name: str, config: LLMConfig):
        """Initialize the OpenAI agent."""
        super().__init__(name, config)
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            timeout=config.timeout
        )
        
        # Model configuration
        self.model = config.model or "gpt-4"
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        
        logger.info(f"Initialized OpenAI agent with model: {self.model}")
    
    def _get_model_name(self) -> str:
        """Get the model name for this agent."""
        return self.model
    
    def _create_search_prompt(self, query: str) -> str:
        """
        Create an OpenAI-specific prompt for brand search.
        
        Args:
            query: The user's search query
            
        Returns:
            Formatted prompt optimized for OpenAI models
        """
        return f"""
You are a research assistant helping to analyze the competitive landscape for data analytics and business intelligence tools.

User Query: "{query}"

Please provide a comprehensive response that includes:

1. A list of the top companies, tools, and platforms that match this query
2. Brief descriptions of what each company/tool offers
3. Key features and capabilities
4. Target markets and use cases
5. Any notable partnerships or integrations

Focus on being thorough and including both well-known market leaders and emerging players. Include specific company names and be detailed about their offerings.

When discussing companies, please mention their full names and any alternative names they might be known by. Include both established enterprises and innovative startups in your response.

Please provide specific, factual information rather than general statements.
"""
    
    async def _make_llm_request(self, query: str) -> str:
        """
        Make a request to OpenAI's API.
        
        Args:
            query: The search query to process
            
        Returns:
            Raw response from OpenAI
            
        Raises:
            Exception: If the API request fails
        """
        try:
            prompt = self._create_search_prompt(query)
            
            logger.debug(f"Making OpenAI request for query: {query}")
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a knowledgeable research assistant specializing in data analytics, business intelligence, and enterprise software. Provide detailed, accurate information about companies and tools in these sectors."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.config.timeout
            )
            
            logger.debug(f"OpenAI response received: {response}")
            
            if not response.choices:
                raise Exception("No response choices returned from OpenAI")
            
            content = response.choices[0].message.content
            logger.debug(f"OpenAI content: '{content}'")
            
            if not content or content.strip() == "":
                # Try a simpler prompt if the first one returns empty
                logger.warning("Empty content received, trying simpler prompt")
                simple_response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": f"List the top companies for: {query}"
                        }
                    ],
                    max_tokens=500,
                    temperature=0.1
                )
                
                if simple_response.choices and simple_response.choices[0].message.content:
                    content = simple_response.choices[0].message.content
                    logger.debug(f"Simple prompt content: '{content}'")
                else:
                    raise Exception("Both prompts returned empty content from OpenAI")
            
            logger.debug(f"OpenAI response length: {len(content)} characters")
            
            return content.strip()
            
        except openai.APITimeoutError as e:
            raise Exception(f"OpenAI API timeout: {str(e)}")
        except openai.APIError as e:
            raise Exception(f"OpenAI API error: {str(e)}")
        except openai.RateLimitError as e:
            raise Exception(f"OpenAI rate limit exceeded: {str(e)}")
        except openai.AuthenticationError as e:
            raise Exception(f"OpenAI authentication error: {str(e)}")
        except Exception as e:
            if "OpenAI" not in str(e):
                raise Exception(f"Unexpected OpenAI error: {str(e)}")
            raise
    
    def _extract_token_usage(self, response_data: Any) -> Optional[Dict[str, int]]:
        """Extract token usage from OpenAI response."""
        if hasattr(response_data, 'usage') and response_data.usage:
            return {
                'prompt_tokens': response_data.usage.prompt_tokens,
                'completion_tokens': response_data.usage.completion_tokens,
                'total_tokens': response_data.usage.total_tokens
            }
        return None
    
    def _estimate_cost(self, token_usage: Optional[Dict[str, int]]) -> Optional[float]:
        """
        Estimate cost for OpenAI API usage.
        
        Note: Prices are approximate and may change. Update regularly.
        """
        if not token_usage:
            return None
        
        # Approximate pricing for GPT-4 (as of 2024)
        # Update these values based on current OpenAI pricing
        price_per_1k_input = 0.03   # $0.03 per 1K input tokens
        price_per_1k_output = 0.06  # $0.06 per 1K output tokens
        
        if self.model.startswith("gpt-3.5"):
            price_per_1k_input = 0.0015   # $0.0015 per 1K input tokens
            price_per_1k_output = 0.002   # $0.002 per 1K output tokens
        
        input_cost = (token_usage.get('prompt_tokens', 0) / 1000) * price_per_1k_input
        output_cost = (token_usage.get('completion_tokens', 0) / 1000) * price_per_1k_output
        
        return input_cost + output_cost
    
    async def test_connection(self) -> bool:
        """
        Test the connection to OpenAI API.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "Hello, this is a connection test."}
                ],
                max_tokens=10,
                temperature=0
            )
            
            return bool(response.choices and response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        return {
            'provider': 'OpenAI',
            'model': self.model,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'timeout': self.config.timeout
        }

# Utility function for creating OpenAI agents
def create_openai_agent(name: str = "openai", api_key: str = None, model: str = "gpt-4") -> OpenAIAgent:
    """
    Utility function to create an OpenAI agent with default configuration.
    
    Args:
        name: Name for the agent
        api_key: OpenAI API key (uses environment variable if None)
        model: Model to use
        
    Returns:
        Configured OpenAI agent
    """
    from src.config.settings import get_settings
    
    settings = get_settings()
    config = LLMConfig(
        name=name,
        api_key=api_key or settings.openai_api_key,
        model=model,
        max_tokens=1000,
        temperature=0.1,
        timeout=30
    )
    
    return OpenAIAgent(name, config)