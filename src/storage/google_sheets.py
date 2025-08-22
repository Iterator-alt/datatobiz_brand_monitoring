"""
Google Sheets Integration

This module provides comprehensive Google Sheets integration for storing
brand monitoring results with proper error handling and retry logic.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import gspread
from gspread.exceptions import APIError, SpreadsheetNotFound, WorksheetNotFound
from oauth2client.service_account import ServiceAccountCredentials

from src.workflow.state import WorkflowState, QueryState, AgentResult
from src.config.settings import GoogleSheetsConfig, get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

class GoogleSheetsManager:
    """
    Manager class for Google Sheets operations with comprehensive error handling.
    
    Handles authentication, sheet operations, and data formatting for brand monitoring results.
    """
    
    def __init__(self, config: GoogleSheetsConfig = None):
        """Initialize the Google Sheets manager."""
        self.config = config or get_settings().google_sheets
        self._client = None
        self._spreadsheet = None
        self._worksheet = None
        
        # Column definitions for Stage 1
        self.stage1_columns = [
            "Query",
            "Model_Name", 
            "Found_Y/N",
            "Timestamp",
            "Confidence",
            "Execution_Time",
            "Error_Message"
        ]
        
        # Additional columns for Stage 2 (ready for future expansion)
        self.stage2_columns = [
            "Ranking_Position",
            "Ranking_Context",
            "Raw_Response_Length",
            "Token_Usage",
            "Cost_Estimate",
            "Retry_Count"
        ]
    
    async def initialize(self) -> bool:
        """
        Initialize the Google Sheets connection and setup.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Setup authentication
            if not await self._authenticate():
                return False
            
            # Open spreadsheet
            if not await self._open_spreadsheet():
                return False
            
            # Setup worksheet
            if not await self._setup_worksheet():
                return False
            
            logger.info("Google Sheets manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Sheets manager: {str(e)}")
            return False
    
    async def _authenticate(self) -> bool:
        """Authenticate with Google Sheets API."""
        try:
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
            
            credentials = ServiceAccountCredentials.from_json_keyfile_name(
                self.config.credentials_file, 
                scope
            )
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self._client = await loop.run_in_executor(
                None, 
                gspread.authorize, 
                credentials
            )
            
            logger.debug("Google Sheets authentication successful")
            return True
            
        except FileNotFoundError:
            logger.error(f"Credentials file not found: {self.config.credentials_file}")
            return False
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return False
    
    async def _open_spreadsheet(self) -> bool:
        """Open the target spreadsheet."""
        try:
            loop = asyncio.get_event_loop()
            self._spreadsheet = await loop.run_in_executor(
                None,
                self._client.open_by_key,
                self.config.spreadsheet_id
            )
            
            logger.debug(f"Opened spreadsheet: {self._spreadsheet.title}")
            return True
            
        except SpreadsheetNotFound:
            logger.error(f"Spreadsheet not found: {self.config.spreadsheet_id}")
            return False
        except Exception as e:
            logger.error(f"Failed to open spreadsheet: {str(e)}")
            return False
    
    async def _setup_worksheet(self) -> bool:
        """Setup or create the target worksheet."""
        try:
            loop = asyncio.get_event_loop()
            
            logger.info(f"Setting up worksheet: {self.config.worksheet_name}")
            logger.info(f"Spreadsheet ID: {self.config.spreadsheet_id}")
            
            # List all available worksheets for debugging
            try:
                all_worksheets = await loop.run_in_executor(
                    None,
                    self._spreadsheet.worksheets
                )
                worksheet_names = [ws.title for ws in all_worksheets]
                logger.info(f"Available worksheets: {worksheet_names}")
            except Exception as e:
                logger.warning(f"Could not list worksheets: {str(e)}")
            
            # Try to open existing worksheet
            try:
                logger.info("Attempting to open existing worksheet...")
                self._worksheet = await loop.run_in_executor(
                    None,
                    self._spreadsheet.worksheet,
                    self.config.worksheet_name
                )
                logger.info(f"Successfully opened existing worksheet: {self.config.worksheet_name}")
                
            except WorksheetNotFound:
                # Create new worksheet
                logger.info("Worksheet not found, creating new worksheet...")
                self._worksheet = await loop.run_in_executor(
                    None,
                    self._spreadsheet.add_worksheet,
                    self.config.worksheet_name,
                    1000,  # rows
                    len(self.stage1_columns) + len(self.stage2_columns)  # columns
                )
                logger.info(f"Successfully created new worksheet: {self.config.worksheet_name}")
            
            # Setup headers if needed
            logger.info("Setting up headers...")
            await self._setup_headers()
            logger.info("Headers setup completed")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup worksheet: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    async def _setup_headers(self):
        """Setup column headers in the worksheet."""
        try:
            loop = asyncio.get_event_loop()
            
            # Check if headers already exist
            existing_headers = await loop.run_in_executor(
                None,
                lambda: self._worksheet.row_values(1)
            )
            
            current_columns = self.stage1_columns.copy()
            
            # Add Stage 2 columns if enabled
            settings = get_settings()
            if settings.enable_ranking_detection:
                current_columns.extend(self.stage2_columns)
            
            if not existing_headers or existing_headers != current_columns:
                # Update headers
                await loop.run_in_executor(
                    None,
                    lambda: self._worksheet.update('A1', [current_columns])
                )
                
                # Format headers (bold)
                await loop.run_in_executor(
                    None,
                    lambda: self._worksheet.format('A1:Z1', {'textFormat': {'bold': True}})
                )
                
                logger.debug("Headers setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup headers: {str(e)}")
            raise
    
    async def store_results(self, workflow_state: WorkflowState) -> bool:
        """
        Store workflow results to Google Sheets.
        
        Args:
            workflow_state: Complete workflow state with results
            
        Returns:
            True if storage successful, False otherwise
        """
        try:
            if not self._worksheet:
                logger.error("Worksheet not initialized")
                return False
            
            # Prepare data rows
            rows_to_add = []
            
            for query, query_state in workflow_state.query_states.items():
                for agent_name, agent_result in query_state.agent_results.items():
                    row = self._format_result_row(query, agent_result)
                    rows_to_add.append(row)
            
            if not rows_to_add:
                logger.warning("No results to store")
                return True
            
            # Add rows to sheet
            await self._append_rows(rows_to_add)
            
            logger.info(f"Stored {len(rows_to_add)} result rows to Google Sheets")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store results: {str(e)}")
            return False
    
    def _format_result_row(self, query: str, agent_result: AgentResult) -> List[str]:
        """Format a single result into a spreadsheet row."""
        # Base columns (Stage 1)
        row = [
            query,
            agent_result.model_name,
            "Y" if agent_result.brand_detection and agent_result.brand_detection.found else "N",
            agent_result.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            f"{agent_result.brand_detection.confidence:.3f}" if agent_result.brand_detection else "0.000",
            f"{agent_result.execution_time:.3f}" if agent_result.execution_time else "",
            agent_result.error_message or ""
        ]
        
        # Stage 2 columns (if enabled)
        settings = get_settings()
        if settings.enable_ranking_detection and agent_result.brand_detection:
            row.extend([
                str(agent_result.brand_detection.ranking_position or ""),
                agent_result.brand_detection.ranking_context or "",
                str(len(agent_result.raw_response)) if agent_result.raw_response else "",
                str(agent_result.token_usage) if agent_result.token_usage else "",
                f"{agent_result.cost_estimate:.6f}" if agent_result.cost_estimate else "",
                str(agent_result.retry_count)
            ])
        
        return row
    
    async def _append_rows(self, rows: List[List[str]], batch_size: int = 100):
        """
        Append multiple rows to the worksheet with batching.
        
        Args:
            rows: List of rows to append
            batch_size: Number of rows to process per batch
        """
        loop = asyncio.get_event_loop()
        
        # Process in batches to avoid API limits
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            
            try:
                await loop.run_in_executor(
                    None,
                    lambda: self._worksheet.append_rows(batch)
                )
                
                logger.debug(f"Appended batch of {len(batch)} rows")
                
                # Small delay to respect API limits
                await asyncio.sleep(0.1)
                
            except APIError as e:
                if "quota" in str(e).lower():
                    logger.warning("Google Sheets API quota exceeded, waiting...")
                    await asyncio.sleep(60)  # Wait 1 minute
                    # Retry the batch
                    await loop.run_in_executor(
                        None,
                        lambda: self._worksheet.append_rows(batch)
                    )
                else:
                    raise
    
    async def store_single_result(self, query: str, agent_result: AgentResult) -> bool:
        """
        Store a single agent result.
        
        Args:
            query: The search query
            agent_result: Result from an agent
            
        Returns:
            True if storage successful, False otherwise
        """
        try:
            if not self._worksheet:
                logger.error("Worksheet not initialized")
                return False
            
            logger.info(f"Attempting to store result for query: {query}")
            logger.info(f"Agent result status: {agent_result.status}")
            
            row = self._format_result_row(query, agent_result)
            logger.info(f"Formatted row: {row}")
            
            loop = asyncio.get_event_loop()
            
            # Try to append the row
            logger.info("Appending row to Google Sheets...")
            await loop.run_in_executor(
                None,
                lambda: self._worksheet.append_row(row)
            )
            
            logger.info(f"Successfully stored single result for query: {query}, agent: {agent_result.agent_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store single result: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    async def get_historical_data(
        self, 
        days_back: int = 30,
        query_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve historical data from the sheet.
        
        Args:
            days_back: Number of days to look back
            query_filter: Optional filter for specific queries
            
        Returns:
            List of historical records
        """
        try:
            if not self._worksheet:
                logger.error("Worksheet not initialized")
                return []
            
            loop = asyncio.get_event_loop()
            
            # Get all records
            records = await loop.run_in_executor(
                None,
                self._worksheet.get_all_records
            )
            
            # Filter by date if specified
            if days_back > 0:
                cutoff_date = datetime.now() - timedelta(days=days_back)
                
                filtered_records = []
                for record in records:
                    try:
                        record_date = datetime.strptime(record.get('Timestamp', ''), "%Y-%m-%d %H:%M:%S")
                        if record_date >= cutoff_date:
                            filtered_records.append(record)
                    except ValueError:
                        continue  # Skip records with invalid timestamps
                
                records = filtered_records
            
            # Filter by query if specified
            if query_filter:
                records = [r for r in records if query_filter.lower() in r.get('Query', '').lower()]
            
            logger.debug(f"Retrieved {len(records)} historical records")
            return records
            
        except Exception as e:
            logger.error(f"Failed to retrieve historical data: {str(e)}")
            return []
    
    async def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from stored data."""
        try:
            records = await self.get_historical_data(days_back=0)  # All data
            
            if not records:
                return {}
            
            total_queries = len(set(r.get('Query', '') for r in records))
            total_results = len(records)
            brand_mentions = len([r for r in records if r.get('Found_Y/N') == 'Y'])
            
            models = list(set(r.get('Model_Name', '') for r in records))
            
            return {
                'total_unique_queries': total_queries,
                'total_results': total_results,
                'brand_mentions_found': brand_mentions,
                'detection_rate': brand_mentions / total_results if total_results > 0 else 0,
                'models_used': models,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get summary stats: {str(e)}")
            return {}
    
    async def cleanup(self):
        """Cleanup resources."""
        # Google Sheets client doesn't require explicit cleanup
        self._client = None
        self._spreadsheet = None
        self._worksheet = None
        logger.debug("Google Sheets manager cleaned up")

# Utility functions
async def create_sheets_manager(config: GoogleSheetsConfig = None) -> GoogleSheetsManager:
    """Create and initialize a Google Sheets manager."""
    manager = GoogleSheetsManager(config)
    success = await manager.initialize()
    
    if not success:
        raise Exception("Failed to initialize Google Sheets manager")
    
    return manager

async def store_workflow_results(workflow_state: WorkflowState, config: GoogleSheetsConfig = None) -> bool:
    """Utility function to store workflow results."""
    manager = GoogleSheetsManager(config)
    
    try:
        if not await manager.initialize():
            return False
        
        return await manager.store_results(workflow_state)
        
    finally:
        await manager.cleanup()