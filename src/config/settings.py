"""
Configuration Management for DataTobiz Brand Monitoring System

This module handles all configuration settings, API keys, and environment variables
with proper security practices and validation.
"""

import os
import yaml
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMConfig(BaseModel):
    """Configuration for individual LLM providers."""
    
    name: str
    api_key: str
    model: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.1
    timeout: int = 30

class GoogleSheetsConfig(BaseModel):
    """Configuration for Google Sheets integration."""
    
    credentials_file: str = Field(default="credentials.json")
    spreadsheet_id: str = ""
    worksheet_name: str = "Brand_Monitoring"

class BrandConfig(BaseModel):
    """Configuration for brand detection settings."""
    
    target_brand: str = "DataTobiz"
    brand_variations: List[str] = ["DataTobiz", "Data Tobiz", "data tobiz", "DATATOBIZ"]
    case_sensitive: bool = False
    partial_match: bool = True
    
    @field_validator('brand_variations')
    @classmethod
    def validate_variations(cls, v, info):
        if info.data.get('target_brand') and info.data['target_brand'] not in v:
            v.append(info.data['target_brand'])
        return v

class WorkflowConfig(BaseModel):
    """Configuration for workflow execution."""
    
    max_retries: int = 3
    retry_delay: float = 1.0
    parallel_execution: bool = True
    timeout_per_agent: int = 60
    log_level: str = "INFO"

class Settings(BaseSettings):
    """Main settings class that aggregates all configurations."""
    
    # API Keys - loaded from environment variables
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    perplexity_api_key: str = Field(default="", env="PERPLEXITY_API_KEY")
    
    # Google Sheets configuration
    google_sheets: GoogleSheetsConfig = Field(default_factory=GoogleSheetsConfig)
    
    # Brand detection configuration
    brand: BrandConfig = Field(default_factory=BrandConfig)
    
    # Workflow configuration
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    
    # LLM Configurations
    llm_configs: Dict[str, LLMConfig] = Field(default_factory=dict)
    
    # Stage 2 Preparation
    enable_ranking_detection: bool = False
    ranking_keywords: List[str] = ["first", "top", "best", "leading", "number one", "#1"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"
    
    def __init__(self, config_file: str = "config.yaml", **kwargs):
        """Initialize settings from config file and environment variables."""
        
        # Load from YAML config file if it exists
        config_data = {}
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f) or {}
        
        # Start with any llm_configs from YAML (raw dicts)
        merged_llm_configs: Dict[str, dict] = dict(config_data.get("llm_configs", {}))
        
        # OpenAI config from environment overrides/augments YAML
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key:
            yaml_openai = merged_llm_configs.get("openai", {}) or {}
            merged_llm_configs["openai"] = {
                "name": "openai",
                "api_key": openai_key,
                "model": yaml_openai.get("model", "gpt-4"),
                "max_tokens": yaml_openai.get("max_tokens", 1000),
                "temperature": yaml_openai.get("temperature", 0.1),
                "timeout": yaml_openai.get("timeout", 30),
            }
        elif "openai" in merged_llm_configs and merged_llm_configs["openai"].get("api_key"):
            # Use API key from YAML config if no environment variable
            yaml_openai = merged_llm_configs["openai"]
            merged_llm_configs["openai"] = {
                "name": "openai",
                "api_key": yaml_openai["api_key"],
                "model": yaml_openai.get("model", "gpt-4"),
                "max_tokens": yaml_openai.get("max_tokens", 1000),
                "temperature": yaml_openai.get("temperature", 0.1),
                "timeout": yaml_openai.get("timeout", 30),
            }
        
        # Perplexity config from environment overrides/augments YAML
        perplexity_key = os.getenv("PERPLEXITY_API_KEY", "")
        if perplexity_key:
            yaml_ppx = merged_llm_configs.get("perplexity", {}) or {}
            merged_llm_configs["perplexity"] = {
                "name": "perplexity",
                "api_key": perplexity_key,
                "model": yaml_ppx.get("model", "llama-3.1-sonar-small-128k-online"),
                "max_tokens": yaml_ppx.get("max_tokens", 1000),
                "temperature": yaml_ppx.get("temperature", 0.1),
                "timeout": yaml_ppx.get("timeout", 30),
            }
        
        # Pull Google Sheets config with fallback to YAML when env is missing
        yaml_gs = config_data.get("google_sheets", {}) or {}
        spreadsheet_id = os.getenv("GOOGLE_SPREADSHEET_ID") or yaml_gs.get("spreadsheet_id", "")
        worksheet_name = yaml_gs.get("worksheet_name", "Brand_Monitoring")
        credentials_file = yaml_gs.get("credentials_file", "credentials.json")
        
        google_sheets_config = GoogleSheetsConfig(
            spreadsheet_id=spreadsheet_id,
            worksheet_name=worksheet_name,
            credentials_file=credentials_file
        )
        
        # Merge config data with defaults
        merged_config = {
            **config_data,
            **kwargs,
            "llm_configs": merged_llm_configs,
            "google_sheets": google_sheets_config,
            "openai_api_key": openai_key,
            "perplexity_api_key": perplexity_key,
        }
        
        super().__init__(**merged_config)
    
    def validate_configuration(self) -> List[str]:
        """Validate all configuration settings and return any errors."""
        errors = []
        
        # Check API keys - at least one required
        if not self.openai_api_key and not self.perplexity_api_key:
            errors.append("At least one API key (OpenAI or Perplexity) is required")
        
        # Google Sheets configuration is optional; log-only issues are not fatal
        # (Storage will be disabled if configuration is incomplete.)
        
        # Check brand configuration
        if not self.brand.target_brand:
            errors.append("Target brand name is missing")
        
        return errors
    
    def get_llm_config(self, llm_name: str) -> Optional[LLMConfig]:
        """Get configuration for a specific LLM."""
        return self.llm_configs.get(llm_name)
    
    def add_llm_config(self, name: str, config: LLMConfig):
        """Add a new LLM configuration (for Stage 2 expansion)."""
        self.llm_configs[name] = config

# Global settings instance
settings = None

def get_settings(config_file: str = "config.yaml") -> Settings:
    """Get or create the global settings instance."""
    global settings
    if settings is None:
        settings = Settings(config_file=config_file)
    return settings


def reload_settings(config_file: str = "config.yaml") -> Settings:
    """Reload settings from configuration file."""
    global settings
    settings = Settings(config_file=config_file)
    return settings