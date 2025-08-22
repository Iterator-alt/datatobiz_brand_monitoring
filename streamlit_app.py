"""
DataTobiz Brand Monitoring System - Streamlit Web Application

This module provides a comprehensive web interface for the brand monitoring system
using Streamlit, offering both interactive monitoring and historical analysis.
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import sys
from pathlib import Path
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import BrandMonitoringAPI
from src.config.settings import get_settings
from src.utils.logger import setup_logging

# Page configuration
st.set_page_config(
    page_title="DataTobiz Brand Monitoring",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .error-metric {
        border-left-color: #dc3545;
    }
    .stButton > button {
        width: 100%;
        margin-top: 1rem;
    }
    .result-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .found-badge {
        background-color: #28a745;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
    }
    .not-found-badge {
        background-color: #dc3545;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitBrandMonitor:
    """Main class for the Streamlit brand monitoring application."""
    
    def __init__(self):
        """Initialize the Streamlit application."""
        self.api = None
        self.initialized = False
        
        # Setup logging
        setup_logging(log_level="INFO")
        
        # Initialize session state
        if 'monitoring_results' not in st.session_state:
            st.session_state.monitoring_results = None
        if 'historical_data' not in st.session_state:
            st.session_state.historical_data = None
    
    async def initialize_api(self):
        """Initialize the brand monitoring API."""
        if not self.initialized:
            try:
                with st.spinner("Initializing Brand Monitoring System..."):
                    self.api = BrandMonitoringAPI("config.yaml")
                    success = await self.api.initialize()
                    
                    if success:
                        self.initialized = True
                        st.success("✅ System initialized successfully!")
                        return True
                    else:
                        st.error("❌ Failed to initialize system")
                        return False
            except Exception as e:
                st.error(f"❌ Initialization error: {str(e)}")
                return False
        return True
    
    def render_header(self):
        """Render the main header."""
        st.markdown('<h1 class="main-header">🔍 DataTobiz Brand Monitoring System</h1>', unsafe_allow_html=True)
        st.markdown("---")
    
    def render_sidebar(self):
        """Render the sidebar with navigation and settings."""
        st.sidebar.title("🎛️ Navigation")
        
        # Page selection
        page = st.sidebar.selectbox(
            "Choose a page:",
            ["🏠 Dashboard", "🔍 Brand Monitoring", "📊 Historical Analysis", "⚙️ Settings", "🔧 System Status"]
        )
        
        # Quick actions
        st.sidebar.markdown("### 🚀 Quick Actions")
        
        if st.sidebar.button("🔄 Refresh System"):
            st.session_state.monitoring_results = None
            st.session_state.historical_data = None
            st.rerun()
        
        if st.sidebar.button("🧪 Test Connections"):
            st.session_state.test_connections = True
        
        # System info
        st.sidebar.markdown("### ℹ️ System Info")
        try:
            settings = get_settings()
            st.sidebar.info(f"**Target Brand:** {settings.brand.target_brand}")
            st.sidebar.info(f"**Models:** {', '.join(settings.llm_configs.keys())}")
        except:
            st.sidebar.warning("Configuration not loaded")
        
        return page
    
    async def render_dashboard(self):
        """Render the main dashboard page."""
        st.header("📊 Dashboard")
        
        # Initialize API if needed
        if not await self.initialize_api():
            return
        
        # Get system status
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card success-metric">', unsafe_allow_html=True)
            st.metric("System Status", "🟢 Online")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Available Models", len(get_settings().llm_configs))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Target Brand", get_settings().brand.target_brand)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Last Updated", datetime.now().strftime("%H:%M"))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick monitoring section
        st.subheader("🚀 Quick Brand Monitoring")
        
        # Sample queries
        sample_queries = [
            "best data analytics companies 2024",
            "top business intelligence tools",
            "leading data visualization software",
            "enterprise analytics platforms"
        ]
        
        selected_queries = st.multiselect(
            "Select queries to monitor:",
            sample_queries,
            default=sample_queries[:2]
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            custom_query = st.text_input(
                "Or enter a custom query:",
                placeholder="e.g., 'data science consulting companies'"
            )
        
        with col2:
            processing_mode = st.selectbox(
                "Processing Mode:",
                ["parallel", "sequential"],
                help="Parallel: faster but uses more resources. Sequential: slower but more reliable."
            )
        
        if st.button("🔍 Start Monitoring", type="primary"):
            if selected_queries or custom_query:
                queries = selected_queries.copy()
                if custom_query:
                    queries.append(custom_query)
                
                await self.run_monitoring(queries, processing_mode)
            else:
                st.warning("Please select at least one query or enter a custom query.")
        
        # Recent results
        if st.session_state.monitoring_results:
            st.subheader("📋 Recent Results")
            self.display_results(st.session_state.monitoring_results)
    
    async def render_monitoring_page(self):
        """Render the dedicated monitoring page."""
        st.header("🔍 Brand Monitoring")
        
        # Initialize API if needed
        if not await self.initialize_api():
            return
        
        # Query input section
        st.subheader("📝 Query Configuration")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Multiple query input
            queries_text = st.text_area(
                "Enter search queries (one per line):",
                height=150,
                placeholder="best data analytics companies\ntop business intelligence tools\nleading data visualization software"
            )
        
        with col2:
            st.markdown("### ⚙️ Settings")
            
            processing_mode = st.selectbox(
                "Processing Mode:",
                ["parallel", "sequential"],
                help="Parallel: faster but uses more resources. Sequential: slower but more reliable."
            )
            
            output_format = st.selectbox(
                "Output Format:",
                ["summary", "detailed", "json"],
                help="Summary: overview only. Detailed: full agent results. JSON: raw data."
            )
            
            enable_ranking = st.checkbox(
                "Enable Ranking Detection",
                help="Detect position/ranking information (Stage 2 feature)"
            )
        
        # Convert text to list
        queries = []
        if queries_text:
            queries = [q.strip() for q in queries_text.split('\n') if q.strip()]
        
        # Monitoring controls
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🔍 Start Monitoring", type="primary", use_container_width=True):
                if queries:
                    await self.run_monitoring(queries, processing_mode, output_format)
                else:
                    st.warning("Please enter at least one query.")
        
        with col2:
            if st.button("🛑 Stop Monitoring", use_container_width=True):
                st.info("Monitoring stopped.")
        
        with col3:
            if st.button("📊 View Results", use_container_width=True):
                if st.session_state.monitoring_results:
                    self.display_detailed_results(st.session_state.monitoring_results)
                else:
                    st.info("No results available. Run monitoring first.")
        
        # Real-time monitoring section
        st.subheader("⏱️ Real-time Monitoring")
        
        if st.button("🔄 Start Real-time Monitoring"):
            await self.start_realtime_monitoring()
    
    async def render_historical_analysis(self):
        """Render the historical analysis page."""
        st.header("📊 Historical Analysis")
        
        # Initialize API if needed
        if not await self.initialize_api():
            return
        
        # Date range selection
        col1, col2 = st.columns(2)
        
        with col1:
            days_back = st.slider(
                "Days to analyze:",
                min_value=1,
                max_value=365,
                value=30,
                help="Number of days to look back for historical data"
            )
        
        with col2:
            if st.button("📈 Load Historical Data"):
                await self.load_historical_data(days_back)
        
        # Display historical data
        if st.session_state.historical_data:
            st.subheader("📋 Historical Results")
            
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(st.session_state.historical_data)
            
            if not df.empty:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_queries = len(df['Query'].unique())
                    st.metric("Total Unique Queries", total_queries)
                
                with col2:
                    total_results = len(df)
                    st.metric("Total Results", total_results)
                
                with col3:
                    brand_mentions = len(df[df['Found_Y/N'] == 'Y'])
                    st.metric("Brand Mentions", brand_mentions)
                
                with col4:
                    detection_rate = brand_mentions / total_results if total_results > 0 else 0
                    st.metric("Detection Rate", f"{detection_rate:.1%}")
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Detection rate over time
                    df['Date'] = pd.to_datetime(df['Timestamp']).dt.date
                    daily_stats = df.groupby('Date').agg({
                        'Found_Y/N': lambda x: (x == 'Y').sum() / len(x)
                    }).reset_index()
                    
                    fig = px.line(daily_stats, x='Date', y='Found_Y/N', 
                                title='Detection Rate Over Time')
                    fig.update_layout(yaxis_title='Detection Rate')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Model performance
                    model_stats = df.groupby('Model_Name').agg({
                        'Found_Y/N': lambda x: (x == 'Y').sum() / len(x),
                        'Execution_Time': 'mean'
                    }).reset_index()
                    
                    fig = px.bar(model_stats, x='Model_Name', y='Found_Y/N',
                               title='Model Performance Comparison')
                    fig.update_layout(yaxis_title='Detection Rate')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed data table
                st.subheader("📋 Detailed Data")
                st.dataframe(df, use_container_width=True)
                
                # Export options
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"brand_monitoring_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    json_data = df.to_json(orient='records', indent=2)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name=f"brand_monitoring_data_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
    
    async def render_settings_page(self):
        """Render the settings page."""
        st.header("⚙️ Settings")
        
        try:
            settings = get_settings()
            
            # Brand configuration
            st.subheader("🎯 Brand Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                target_brand = st.text_input(
                    "Target Brand:",
                    value=settings.brand.target_brand,
                    help="Primary brand name to search for"
                )
            
            with col2:
                case_sensitive = st.checkbox(
                    "Case Sensitive",
                    value=settings.brand.case_sensitive,
                    help="Whether brand detection should be case sensitive"
                )
            
            brand_variations = st.text_area(
                "Brand Variations (one per line):",
                value='\n'.join(settings.brand.brand_variations),
                help="Different ways the brand name might appear"
            )
            
            # Workflow configuration
            st.subheader("⚙️ Workflow Configuration")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                max_retries = st.number_input(
                    "Max Retries:",
                    min_value=1,
                    max_value=10,
                    value=settings.workflow.max_retries
                )
            
            with col2:
                retry_delay = st.number_input(
                    "Retry Delay (seconds):",
                    min_value=0.1,
                    max_value=10.0,
                    value=settings.workflow.retry_delay,
                    step=0.1
                )
            
            with col3:
                timeout_per_agent = st.number_input(
                    "Timeout per Agent (seconds):",
                    min_value=10,
                    max_value=300,
                    value=settings.workflow.timeout_per_agent
                )
            
            # LLM Configuration
            st.subheader("🤖 LLM Configuration")
            
            for model_name, config in settings.llm_configs.items():
                with st.expander(f"📋 {model_name.upper()} Configuration"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        model = st.text_input(
                            f"Model ({model_name}):",
                            value=config.model or "",
                            key=f"model_{model_name}"
                        )
                    
                    with col2:
                        max_tokens = st.number_input(
                            f"Max Tokens ({model_name}):",
                            min_value=100,
                            max_value=4000,
                            value=config.max_tokens,
                            key=f"max_tokens_{model_name}"
                        )
                    
                    temperature = st.slider(
                        f"Temperature ({model_name}):",
                        min_value=0.0,
                        max_value=1.0,
                        value=config.temperature,
                        step=0.1,
                        key=f"temp_{model_name}"
                    )
            
            # Save settings
            if st.button("💾 Save Settings", type="primary"):
                st.success("Settings saved! (Note: Some changes may require restart)")
        
        except Exception as e:
            st.error(f"Error loading settings: {str(e)}")
    
    async def render_system_status(self):
        """Render the system status page."""
        st.header("🔧 System Status")
        
        # Initialize API if needed
        if not await self.initialize_api():
            return
        
        # Connection testing
        st.subheader("🔌 Connection Status")
        
        if st.button("🧪 Test All Connections"):
            with st.spinner("Testing connections..."):
                result = await self.api.test_connections()
                
                if result['success']:
                    st.success("✅ All connections are working!")
                else:
                    st.error(f"❌ Connection test failed: {result.get('error', 'Unknown error')}")
        
        # System statistics
        st.subheader("📊 System Statistics")
        
        try:
            stats_result = await self.api.get_stats()
            
            if stats_result['success'] and stats_result['stats']:
                stats = stats_result['stats']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Results", stats.get('total_results', 0))
                
                with col2:
                    st.metric("Brand Mentions", stats.get('brand_mentions_found', 0))
                
                with col3:
                    detection_rate = stats.get('detection_rate', 0)
                    st.metric("Detection Rate", f"{detection_rate:.1%}")
                
                with col4:
                    models_used = len(stats.get('models_used', []))
                    st.metric("Models Used", models_used)
                
                # Model breakdown
                if stats.get('models_used'):
                    st.subheader("🤖 Model Usage")
                    model_df = pd.DataFrame({
                        'Model': stats['models_used'],
                        'Status': '✅ Active'
                    })
                    st.dataframe(model_df, use_container_width=True)
            else:
                st.info("No historical data available.")
        
        except Exception as e:
            st.error(f"Error loading statistics: {str(e)}")
        
        # Configuration overview
        st.subheader("⚙️ Configuration Overview")
        
        try:
            settings = get_settings()
            
            config_data = {
                'Setting': [
                    'Target Brand',
                    'Brand Variations',
                    'Case Sensitive',
                    'Partial Match',
                    'Max Retries',
                    'Retry Delay',
                    'Parallel Execution',
                    'Timeout per Agent'
                ],
                'Value': [
                    settings.brand.target_brand,
                    ', '.join(settings.brand.brand_variations),
                    str(settings.brand.case_sensitive),
                    str(settings.brand.partial_match),
                    str(settings.workflow.max_retries),
                    f"{settings.workflow.retry_delay}s",
                    str(settings.workflow.parallel_execution),
                    f"{settings.workflow.timeout_per_agent}s"
                ]
            }
            
            config_df = pd.DataFrame(config_data)
            st.dataframe(config_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading configuration: {str(e)}")
    
    async def run_monitoring(self, queries, processing_mode="parallel", output_format="summary"):
        """Run brand monitoring for the given queries."""
        try:
            with st.spinner(f"Running brand monitoring for {len(queries)} queries..."):
                result = await self.api.monitor_queries(queries, processing_mode)
                
                if result['success']:
                    st.session_state.monitoring_results = result
                    st.success(f"✅ Monitoring completed! Found {result['summary']['brand_mentions_found']} brand mentions.")
                    
                    if output_format == "detailed":
                        self.display_detailed_results(result)
                    elif output_format == "json":
                        st.json(result)
                    else:
                        self.display_results(result)
                else:
                    st.error(f"❌ Monitoring failed: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"❌ Error during monitoring: {str(e)}")
    
    def display_results(self, result):
        """Display monitoring results in a summary format."""
        summary = result['summary']
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", summary['total_queries'])
        
        with col2:
            st.metric("Processed", summary['processed'])
        
        with col3:
            st.metric("Brand Mentions", summary['brand_mentions_found'])
        
        with col4:
            st.metric("Success Rate", f"{summary['success_rate']:.1%}")
        
        # Query results
        st.subheader("📋 Query Results")
        
        for query, query_result in result['results'].items():
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{query}**")
                
                with col2:
                    if query_result['found']:
                        st.markdown('<span class="found-badge">🎯 FOUND</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="not-found-badge">❌ NOT FOUND</span>', unsafe_allow_html=True)
                
                with col3:
                    if query_result['found']:
                        st.write(f"Confidence: {query_result['confidence']:.1%}")
                
                # Agent breakdown
                with st.expander("🤖 Agent Details"):
                    for agent_name, agent_result in query_result['agents'].items():
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.write(f"**{agent_name}**")
                        
                        with col2:
                            status_icon = "✅" if agent_result['status'] == 'completed' else "❌"
                            st.write(f"{status_icon} {agent_result['status']}")
                        
                        with col3:
                            if agent_result['found']:
                                st.write("🎯 Found")
                            else:
                                st.write("❌ Not Found")
                        
                        with col4:
                            st.write(f"{agent_result['execution_time']:.2f}s")
    
    def display_detailed_results(self, result):
        """Display detailed monitoring results."""
        st.subheader("📊 Detailed Results")
        
        # Create detailed DataFrame
        detailed_data = []
        
        for query, query_result in result['results'].items():
            for agent_name, agent_result in query_result['agents'].items():
                detailed_data.append({
                    'Query': query,
                    'Agent': agent_name,
                    'Status': agent_result['status'],
                    'Found': agent_result['found'],
                    'Confidence': agent_result['confidence'],
                    'Execution Time': agent_result['execution_time']
                })
        
        if detailed_data:
            df = pd.DataFrame(detailed_data)
            st.dataframe(df, use_container_width=True)
            
            # Performance visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Agent performance comparison
                agent_stats = df.groupby('Agent').agg({
                    'Found': 'sum',
                    'Execution Time': 'mean'
                }).reset_index()
                
                fig = px.bar(agent_stats, x='Agent', y='Found',
                           title='Brand Mentions by Agent')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Execution time comparison
                fig = px.bar(agent_stats, x='Agent', y='Execution Time',
                           title='Average Execution Time by Agent')
                st.plotly_chart(fig, use_container_width=True)
    
    async def load_historical_data(self, days_back):
        """Load historical data from storage."""
        try:
            with st.spinner("Loading historical data..."):
                # This would need to be implemented based on your storage backend
                # For now, we'll show a placeholder
                st.info("Historical data loading would be implemented here based on your storage backend.")
        
        except Exception as e:
            st.error(f"Error loading historical data: {str(e)}")
    
    async def start_realtime_monitoring(self):
        """Start real-time monitoring (placeholder for future implementation)."""
        st.info("Real-time monitoring feature would be implemented here.")
        st.write("This could include:")
        st.write("- Continuous monitoring with periodic checks")
        st.write("- Real-time alerts and notifications")
        st.write("- Live dashboard updates")
        st.write("- Webhook integrations")

def main():
    """Main function to run the Streamlit application."""
    # Create the application instance
    app = StreamlitBrandMonitor()
    
    # Render header
    app.render_header()
    
    # Render sidebar and get page selection
    page = app.render_sidebar()
    
    # Render the selected page
    if page == "🏠 Dashboard":
        asyncio.run(app.render_dashboard())
    elif page == "🔍 Brand Monitoring":
        asyncio.run(app.render_monitoring_page())
    elif page == "📊 Historical Analysis":
        asyncio.run(app.render_historical_analysis())
    elif page == "⚙️ Settings":
        asyncio.run(app.render_settings_page())
    elif page == "🔧 System Status":
        asyncio.run(app.render_system_status())

if __name__ == "__main__":
    main() 