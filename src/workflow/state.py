"""
State Management for LangGraph Workflow

This module defines the state objects that flow through the LangGraph workflow,
ensuring type safety and proper data handling between agents.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

class AgentStatus(str, Enum):
	"""Status enumeration for agent execution."""
	PENDING = "pending"
	RUNNING = "running"
	COMPLETED = "completed"
	FAILED = "failed"
	SKIPPED = "skipped"

class BrandDetectionResult(BaseModel):
	"""Result of brand detection analysis."""
	
	found: bool = Field(description="Whether the brand was detected")
	confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
	matches: List[str] = Field(default_factory=list, description="Exact text matches found")
	context: Optional[str] = Field(default=None, description="Context around the match")
	
	# Stage 2 preparation
	ranking_position: Optional[int] = Field(default=None, description="Position if ranking detected")
	ranking_context: Optional[str] = Field(default=None, description="Context suggesting ranking")

class AgentResult(BaseModel):
	"""Result from an individual agent execution."""
	
	agent_name: str = Field(description="Name of the agent")
	model_name: str = Field(description="LLM model used")
	status: AgentStatus = Field(default=AgentStatus.PENDING)
	
	# Core results
	raw_response: Optional[str] = Field(default=None, description="Raw LLM response")
	brand_detection: Optional[BrandDetectionResult] = Field(default=None)
	
	# Execution metadata
	execution_time: Optional[float] = Field(default=None, description="Execution time in seconds")
	timestamp: datetime = Field(default_factory=datetime.now)
	error_message: Optional[str] = Field(default=None)
	retry_count: int = Field(default=0)
	
	# Additional metadata
	token_usage: Optional[Dict[str, int]] = Field(default=None)
	cost_estimate: Optional[float] = Field(default=None)

class QueryState(BaseModel):
	"""State object representing the processing of a single query."""
	
	# Input data
	query: str = Field(description="The search query to process")
	query_id: Optional[str] = Field(default=None, description="Unique identifier for the query")
	
	# Processing state
	status: AgentStatus = Field(default=AgentStatus.PENDING)
	current_step: str = Field(default="initialization")
	
	# Agent results
	agent_results: Dict[str, AgentResult] = Field(default_factory=dict)
	
	# Aggregated results
	overall_found: bool = Field(default=False, description="Brand found by any agent")
	consensus_confidence: float = Field(default=0.0, description="Aggregated confidence score")
	
	# Execution metadata
	start_time: datetime = Field(default_factory=datetime.now)
	end_time: Optional[datetime] = Field(default=None)
	total_execution_time: Optional[float] = Field(default=None)
	
	# Error handling
	errors: List[str] = Field(default_factory=list)
	warnings: List[str] = Field(default_factory=list)
	
	# Stage 2 preparation
	best_ranking: Optional[int] = Field(default=None, description="Best ranking found across agents")
	ranking_sources: List[str] = Field(default_factory=list, description="Agents that found rankings")

class WorkflowState(BaseModel):
	"""Main state object for the entire workflow execution."""
	
	# Input configuration
	queries: List[str] = Field(description="List of queries to process")
	target_agents: List[str] = Field(default=["openai"], description="Agents to use")
	
	# Processing state
	current_query_index: int = Field(default=0)
	processing_mode: str = Field(default="parallel", description="sequential or parallel")
	
	# Results tracking
	query_states: Dict[str, QueryState] = Field(default_factory=dict)
	completed_queries: List[str] = Field(default_factory=list)
	failed_queries: List[str] = Field(default_factory=list)
	
	# Overall execution metadata
	workflow_id: str = Field(description="Unique identifier for this workflow execution")
	start_time: datetime = Field(default_factory=datetime.now)
	end_time: Optional[datetime] = Field(default=None)
	
	# Summary statistics
	total_queries: int = Field(default=0)
	successful_queries: int = Field(default=0)
	failed_queries_count: int = Field(default=0)
	total_brand_mentions: int = Field(default=0)
	
	# Storage and output
	storage_results: Dict[str, Any] = Field(default_factory=dict, description="Results from storage operations")
	output_file_paths: List[str] = Field(default_factory=list)
	
	# Configuration and context
	config_snapshot: Optional[Dict[str, Any]] = Field(default=None, description="Configuration used")
	
	def add_query_state(self, query: str, query_state: QueryState):
		"""Add a query state to the workflow."""
		self.query_states[query] = query_state
		
	def get_query_state(self, query: str) -> Optional[QueryState]:
		"""Get the state for a specific query."""
		return self.query_states.get(query)
	
	def mark_query_completed(self, query: str):
		"""Mark a query as completed."""
		if query not in self.completed_queries:
			self.completed_queries.append(query)
			self.successful_queries += 1
	
	def mark_query_failed(self, query: str, error: str):
		"""Mark a query as failed."""
		if query not in self.failed_queries:
			self.failed_queries.append(query)
			self.failed_queries_count += 1
			
		# Add error to query state if it exists
		query_state = self.get_query_state(query)
		if query_state:
			query_state.errors.append(error)
	
	def update_summary_stats(self):
		"""Update summary statistics based on current state."""
		self.total_queries = len(self.queries)
		self.successful_queries = len(self.completed_queries)
		self.failed_queries_count = len(self.failed_queries)
		
		# Count brand mentions
		self.total_brand_mentions = sum(
			1 for query_state in self.query_states.values()
			if query_state.overall_found
		)
	
	def get_progress_summary(self) -> Dict[str, Any]:
		"""Get a summary of current progress."""
		processed = len(self.completed_queries) + len(self.failed_queries)
		
		return {
			"total_queries": self.total_queries,
			"processed": processed,
			"remaining": self.total_queries - processed,
			"success_rate": self.successful_queries / max(processed, 1),
			"brand_mentions_found": self.total_brand_mentions,
			"current_query": self.queries[self.current_query_index] if self.current_query_index < len(self.queries) else None
		}
	
	def is_complete(self) -> bool:
		"""Check if the workflow is complete."""
		return len(self.completed_queries) + len(self.failed_queries) >= len(self.queries)

# Type aliases for compatibility
WorkflowStateDict = Dict[str, Any]


def state_to_dict(state: WorkflowState) -> WorkflowStateDict:
	"""Convert WorkflowState to dictionary."""
	return state.model_dump()


def dict_to_state(state_dict: WorkflowStateDict) -> WorkflowState:
	"""Convert dictionary back to WorkflowState."""
	return WorkflowState.model_validate(state_dict)

# State update utilities

def update_agent_result(state: WorkflowState, query: str, agent_name: str, result: AgentResult) -> WorkflowState:
	"""Update an agent result in the workflow state."""
	query_state = state.get_query_state(query)
	if query_state:
		query_state.agent_results[agent_name] = result
		
		# Update overall found status
		if result.brand_detection and result.brand_detection.found:
			query_state.overall_found = True
			
		# Update consensus confidence
		confidences = [
			r.brand_detection.confidence 
			for r in query_state.agent_results.values() 
			if r.brand_detection
		]
		if confidences:
			query_state.consensus_confidence = sum(confidences) / len(confidences)
	
	return state


def finalize_query_state(state: WorkflowState, query: str) -> WorkflowState:
	"""Finalize the processing of a query state."""
	query_state = state.get_query_state(query)
	if query_state:
		query_state.end_time = datetime.now()
		query_state.total_execution_time = (
			query_state.end_time - query_state.start_time
		).total_seconds()
		query_state.status = AgentStatus.COMPLETED
		
		state.mark_query_completed(query)
	
	return state