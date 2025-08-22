"""
LangGraph Workflow Implementation

This module implements the core LangGraph workflow for orchestrating
multi-agent brand monitoring with proper state management and error handling.
"""

import asyncio
import uuid
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import os

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.workflow.state import (
    WorkflowState, QueryState, AgentResult, AgentStatus,
    state_to_dict, dict_to_state, update_agent_result, finalize_query_state
)
from src.agents.base_agent import BaseAgent, AgentFactory
from src.agents.openai_agent import OpenAIAgent
from src.agents.perplexity_agent import PerplexityAgent
from src.storage.google_sheets import GoogleSheetsManager
from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

class BrandMonitoringWorkflow:
    """
    LangGraph-based workflow for multi-agent brand monitoring.
    
    This class implements a sophisticated workflow that orchestrates multiple
    LLM agents to search for brand mentions with proper state management,
    error handling, and result aggregation.
    """
    
    def __init__(self, config=None):
        """Initialize the workflow with configuration."""
        self.config = config or get_settings()
        self.agents: Dict[str, BaseAgent] = {}
        self.storage_manager: Optional[GoogleSheetsManager] = None
        self.graph = None
        
        # Workflow execution tracking
        self.execution_history: List[Dict[str, Any]] = []
        
        logger.info("Initializing BrandMonitoringWorkflow")
    
    async def initialize(self) -> bool:
        """
        Initialize the workflow components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize agents (non-fatal if none healthy)
            await self._initialize_agents()
            
            # Initialize storage (optional)
            await self._initialize_storage()
            
            # Build the LangGraph
            self._build_graph()
            
            logger.info("Workflow initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Workflow initialization failed: {str(e)}")
            return False
    
    async def _initialize_agents(self) -> bool:
        """Initialize all configured agents."""
        logger.info("Initializing agents...")
        
        success_count = 0
        total_agents = 0
        
        # Initialize OpenAI agent if configured
        if "openai" in self.config.llm_configs:
            total_agents += 1
            try:
                openai_config = self.config.llm_configs["openai"]
                agent = OpenAIAgent("openai", openai_config)
                
                # Always add the agent, even if health check fails (for quota issues)
                self.agents["openai"] = agent
                success_count += 1
                logger.info("OpenAI agent initialized (health check may fail due to quota)")
                    
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI agent: {str(e)}")
        
        # Initialize Perplexity agent if configured
        if "perplexity" in self.config.llm_configs:
            total_agents += 1
            try:
                perplexity_config = self.config.llm_configs["perplexity"]
                agent = PerplexityAgent("perplexity", perplexity_config)
                
                # Test health check
                if await agent.test_connection():
                    self.agents["perplexity"] = agent
                    success_count += 1
                    logger.info("Perplexity agent initialized successfully")
                else:
                    logger.error("Agent perplexity failed health check; skipping")
                    
            except Exception as e:
                logger.error(f"Failed to initialize Perplexity agent: {str(e)}")
        
        if success_count == 0:
            logger.warning("No agents successfully initialized; proceeding without agents")
            return False
        
        logger.info(f"Initialized {success_count}/{total_agents} agents successfully")
        return True
    
    async def _initialize_storage(self) -> bool:
        """Initialize Google Sheets storage manager (optional)."""
        try:
            gs_cfg = self.config.google_sheets
            if not gs_cfg.spreadsheet_id or not os.path.exists(gs_cfg.credentials_file):
                logger.warning("Google Sheets not configured or credentials missing; storage disabled")
                self.storage_manager = None
                return True
            
            self.storage_manager = GoogleSheetsManager(self.config.google_sheets)
            ok = await self.storage_manager.initialize()
            if not ok:
                logger.warning("Google Sheets initialization failed; continuing without storage")
                self.storage_manager = None
            return True
            
        except Exception as e:
            logger.error(f"Storage initialization failed: {str(e)}")
            self.storage_manager = None
            return True
    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        # Create state graph
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("start", self._start_node)
        workflow.add_node("process_query", self._process_query_node)
        workflow.add_node("run_agents_parallel", self._run_agents_parallel_node)
        workflow.add_node("run_agents_sequential", self._run_agents_sequential_node)
        workflow.add_node("aggregate_results", self._aggregate_results_node)
        workflow.add_node("store_results", self._store_results_node)
        workflow.add_node("finalize", self._finalize_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Define edges
        workflow.set_entry_point("start")
        
        workflow.add_edge("start", "process_query")
        
        # Conditional routing based on processing mode
        workflow.add_conditional_edges(
            "process_query",
            self._decide_execution_mode,
            {
                "parallel": "run_agents_parallel",
                "sequential": "run_agents_sequential",
                "error": "handle_error",
                "complete": "finalize"
            }
        )
        
        workflow.add_edge("run_agents_parallel", "aggregate_results")
        workflow.add_edge("run_agents_sequential", "aggregate_results")
        workflow.add_edge("aggregate_results", "store_results")
        workflow.add_edge("store_results", "process_query")  # Loop back for next query
        workflow.add_edge("handle_error", "process_query")   # Continue on error
        workflow.add_edge("finalize", END)
        
        # Compile the graph
        checkpointer = MemorySaver()
        self.graph = workflow.compile(checkpointer=checkpointer)
        
        logger.info("LangGraph workflow compiled successfully")
    
    # Node implementations
    async def _start_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize the workflow state."""
        logger.info("Starting brand monitoring workflow")
        
        workflow_state = dict_to_state(state)
        workflow_state.workflow_id = str(uuid.uuid4())
        workflow_state.start_time = datetime.now()
        
        # Prepare query states
        for i, query in enumerate(workflow_state.queries):
            query_state = QueryState(
                query=query,
                query_id=f"{workflow_state.workflow_id}_{i}"
            )
            workflow_state.add_query_state(query, query_state)
        
        workflow_state.total_queries = len(workflow_state.queries)
        
        logger.info(f"Initialized workflow with {workflow_state.total_queries} queries")
        return state_to_dict(workflow_state)
    
    async def _process_query_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the current query."""
        workflow_state = dict_to_state(state)
        
        # Check if we're done
        if workflow_state.current_query_index >= len(workflow_state.queries):
            workflow_state.end_time = datetime.now()
            return state_to_dict(workflow_state)
        
        current_query = workflow_state.queries[workflow_state.current_query_index]
        logger.info(f"Processing query {workflow_state.current_query_index + 1}/{len(workflow_state.queries)}: {current_query}")
        
        # Update current step
        query_state = workflow_state.get_query_state(current_query)
        if query_state:
            query_state.current_step = "processing"
            query_state.status = AgentStatus.RUNNING
        
        return state_to_dict(workflow_state)
    
    async def _decide_execution_mode(self, state: Dict[str, Any]) -> str:
        """Decide how to execute agents based on workflow state."""
        workflow_state = dict_to_state(state)
        
        # Check if we're done with all queries
        if workflow_state.current_query_index >= len(workflow_state.queries):
            return "complete"
        
        # Check for errors that should halt processing
        if workflow_state.failed_queries_count > len(workflow_state.queries) * 0.5:
            logger.warning("Too many failed queries, entering error handling")
            return "error"
        
        # Decide execution mode
        if workflow_state.processing_mode == "parallel":
            return "parallel"
        else:
            return "sequential"
    
    async def _run_agents_parallel_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run all agents in parallel for the current query."""
        workflow_state = dict_to_state(state)
        current_query = workflow_state.queries[workflow_state.current_query_index]
        
        logger.info(f"Running agents in parallel for query: {current_query}")
        
        # Create tasks for all agents
        tasks = []
        for agent_name, agent in self.agents.items():
            task = asyncio.create_task(
                self._run_single_agent(agent_name, agent, current_query),
                name=f"agent_{agent_name}"
            )
            tasks.append(task)
        
        # Wait for all agents to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        query_state = workflow_state.get_query_state(current_query)
        for i, (agent_name, result) in enumerate(zip(self.agents.keys(), results)):
            if isinstance(result, Exception):
                # Create error result
                error_result = AgentResult(
                    agent_name=agent_name,
                    model_name=self.agents[agent_name]._get_model_name(),
                    status=AgentStatus.FAILED,
                    error_message=str(result)
                )
                query_state.agent_results[agent_name] = error_result
                logger.error(f"Agent {agent_name} failed: {str(result)}")
            else:
                query_state.agent_results[agent_name] = result
                logger.info(f"Agent {agent_name} completed successfully")
        
        return state_to_dict(workflow_state)
    
    async def _run_agents_sequential_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run agents sequentially for the current query."""
        workflow_state = dict_to_state(state)
        current_query = workflow_state.queries[workflow_state.current_query_index]
        
        logger.info(f"Running agents sequentially for query: {current_query}")
        
        query_state = workflow_state.get_query_state(current_query)
        
        # Run agents one by one
        for agent_name, agent in self.agents.items():
            try:
                result = await self._run_single_agent(agent_name, agent, current_query)
                query_state.agent_results[agent_name] = result
                logger.info(f"Agent {agent_name} completed successfully")
                
                # Small delay between agents
                await asyncio.sleep(0.5)
                
            except Exception as e:
                error_result = AgentResult(
                    agent_name=agent_name,
                    model_name=agent._get_model_name(),
                    status=AgentStatus.FAILED,
                    error_message=str(e)
                )
                query_state.agent_results[agent_name] = error_result
                logger.error(f"Agent {agent_name} failed: {str(e)}")
        
        return state_to_dict(workflow_state)
    
    async def _run_single_agent(self, agent_name: str, agent: BaseAgent, query: str) -> AgentResult:
        """Run a single agent for a query."""
        logger.debug(f"Running agent {agent_name} for query: {query}")
        
        try:
            result = await agent.execute(query)
            return result
            
        except Exception as e:
            logger.error(f"Agent {agent_name} execution failed: {str(e)}")
            raise
    
    async def _aggregate_results_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from all agents."""
        workflow_state = dict_to_state(state)
        current_query = workflow_state.queries[workflow_state.current_query_index]
        
        logger.debug(f"Aggregating results for query: {current_query}")
        
        query_state = workflow_state.get_query_state(current_query)
        
        # Calculate overall results
        successful_agents = [
            result for result in query_state.agent_results.values()
            if result.status == AgentStatus.COMPLETED and result.brand_detection
        ]
        
        if successful_agents:
            # Check if any agent found the brand
            query_state.overall_found = any(
                result.brand_detection.found for result in successful_agents
            )
            
            # Calculate consensus confidence
            confidences = [
                result.brand_detection.confidence 
                for result in successful_agents
                if result.brand_detection
            ]
            if confidences:
                query_state.consensus_confidence = sum(confidences) / len(confidences)
            
            # Stage 2 preparation: Find best ranking
            rankings = [
                result.brand_detection.ranking_position
                for result in successful_agents
                if result.brand_detection and result.brand_detection.ranking_position
            ]
            if rankings:
                query_state.best_ranking = min(rankings)  # Lower is better
                query_state.ranking_sources = [
                    result.agent_name for result in successful_agents
                    if result.brand_detection and result.brand_detection.ranking_position
                ]
        
        # Mark query as completed
        finalize_query_state(workflow_state, current_query)
        
        logger.info(
            f"Query aggregation completed. Brand found: {query_state.overall_found}, "
            f"Confidence: {query_state.consensus_confidence:.3f}"
        )
        
        return state_to_dict(workflow_state)
    
    async def _store_results_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Store results to Google Sheets."""
        workflow_state = dict_to_state(state)
        current_query = workflow_state.queries[workflow_state.current_query_index]
        
        logger.debug(f"Storing results for query: {current_query}")
        
        try:
            if self.storage_manager:
                # Store results for current query
                query_state = workflow_state.get_query_state(current_query)
                for agent_name, agent_result in query_state.agent_results.items():
                    success = await self.storage_manager.store_single_result(current_query, agent_result)
                    if not success:
                        logger.warning(f"Failed to store result for agent {agent_name}")
                
                logger.info(f"Results stored for query: {current_query}")
            else:
                logger.warning("Storage manager not available")
        
        except Exception as e:
            logger.error(f"Failed to store results: {str(e)}")
            # Continue execution even if storage fails
        
        # Move to next query
        workflow_state.current_query_index += 1
        
        return state_to_dict(workflow_state)
    
    async def _handle_error_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle workflow errors."""
        workflow_state = dict_to_state(state)
        
        logger.warning("Entering error handling mode")
        
        # Log current state
        progress = workflow_state.get_progress_summary()
        logger.warning(f"Workflow progress: {progress}")
        
        # For now, just continue processing
        # In the future, this could implement recovery strategies
        
        return state_to_dict(workflow_state)
    
    async def _finalize_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize the workflow execution."""
        workflow_state = dict_to_state(state)
        
        logger.info("Finalizing workflow execution")
        
        # Update final statistics
        workflow_state.update_summary_stats()
        
        # Log final summary
        summary = workflow_state.get_progress_summary()
        logger.info(f"Workflow completed. Summary: {summary}")
        
        # Record execution in history
        execution_record = {
            'workflow_id': workflow_state.workflow_id,
            'start_time': workflow_state.start_time,
            'end_time': workflow_state.end_time,
            'total_queries': workflow_state.total_queries,
            'successful_queries': workflow_state.successful_queries,
            'brand_mentions': workflow_state.total_brand_mentions,
            'agents_used': list(self.agents.keys())
        }
        self.execution_history.append(execution_record)
        
        return state_to_dict(workflow_state)
    
    # Public interface methods
    async def execute_workflow(self, queries: List[str], processing_mode: str = "parallel") -> WorkflowState:
        """
        Execute the complete workflow for a list of queries.
        
        Args:
            queries: List of search queries to process
            processing_mode: "parallel" or "sequential" execution
            
        Returns:
            Final workflow state with all results
        """
        if not self.graph:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")
        
        logger.info(f"Starting workflow execution with {len(queries)} queries in {processing_mode} mode")
        
        # Create initial state
        initial_state = WorkflowState(
            queries=queries,
            target_agents=list(self.agents.keys()),
            processing_mode=processing_mode,
            config_snapshot=self.config.model_dump() if hasattr(self.config, 'model_dump') else None,
            workflow_id=str(uuid.uuid4()),
            start_time=datetime.now()
        )
        
        # Execute the graph
        try:
            thread_id = str(uuid.uuid4())
            final_state_dict = await self.graph.ainvoke(
                state_to_dict(initial_state),
                config={"configurable": {"thread_id": thread_id}}
            )
            
            final_state = dict_to_state(final_state_dict)
            
            logger.info("Workflow execution completed successfully")
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            raise
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get the history of workflow executions."""
        return self.execution_history.copy()
    
    def get_agent_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all agents."""
        stats = {}
        for agent_name, agent in self.agents.items():
            stats[agent_name] = agent.get_performance_stats()
        return stats
    
    async def cleanup(self):
        """Clean up workflow resources."""
        logger.info("Cleaning up workflow resources")
        
        # Cleanup agents
        for agent in self.agents.values():
            if hasattr(agent, 'cleanup'):
                await agent.cleanup()
        
        # Cleanup storage
        if self.storage_manager:
            await self.storage_manager.cleanup()
        
        logger.info("Workflow cleanup completed")

# Factory function for easy workflow creation
async def create_workflow(config=None) -> BrandMonitoringWorkflow:
    """
    Create and initialize a brand monitoring workflow.
    
    Args:
        config: Optional configuration object
        
    Returns:
        Initialized workflow ready for execution
    """
    workflow = BrandMonitoringWorkflow(config)
    
    if not await workflow.initialize():
        raise RuntimeError("Failed to initialize workflow")
    
    return workflow