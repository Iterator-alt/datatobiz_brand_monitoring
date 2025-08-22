# 🚀 Quick Start Guide - DataTobiz Brand Monitoring System

## Prerequisites

1. **Python 3.8+** installed on your system
2. **API Keys** for OpenAI and/or Perplexity
3. **Git** (optional, for cloning)

## 🖥️ Windows Setup

### Option 1: Using the Startup Script (Recommended)

1. **Double-click** `start_streamlit.bat` in the project directory
2. The script will automatically:
   - Check Python installation
   - Create virtual environment
   - Install dependencies
   - Start the application

### Option 2: Using PowerShell

1. **Right-click** `start_streamlit.ps1` and select "Run with PowerShell"
2. If you get a security error, run this in PowerShell:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

### Option 3: Manual Setup

1. **Open Command Prompt** in the project directory
2. **Create virtual environment:**
   ```cmd
   python -m venv venv
   ```
3. **Activate virtual environment:**
   ```cmd
   venv\Scripts\activate
   ```
4. **Install dependencies:**
   ```cmd
   pip install -r requirements.txt
   ```
5. **Create environment file:**
   ```cmd
   copy env.template .env
   ```
6. **Edit `.env` file** with your API keys
7. **Start the application:**
   ```cmd
   streamlit run streamlit_app.py
   ```

## 🐧 Linux/Mac Setup

### Option 1: Using the Deployment Script

1. **Make script executable:**
   ```bash
   chmod +x deploy.sh
   ```
2. **Run the script:**
   ```bash
   ./deploy.sh local
   ```

### Option 2: Manual Setup

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   ```
2. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Create environment file:**
   ```bash
   cp env.template .env
   ```
5. **Edit `.env` file** with your API keys
6. **Start the application:**
   ```bash
   streamlit run streamlit_app.py
   ```

## 🐳 Docker Setup

### Prerequisites
- Docker Desktop installed and running
- Docker Compose installed

### Quick Start

1. **Start Docker Desktop**
2. **Open terminal** in the project directory
3. **Create environment file:**
   ```bash
   cp env.template .env
   ```
4. **Edit `.env` file** with your API keys
5. **Build and run:**
   ```bash
   docker-compose up --build -d
   ```
6. **Access the application** at: http://localhost:8501

## 🔑 API Key Setup

### Required API Keys

You need at least one of these API keys:

1. **OpenAI API Key:**
   - Visit: https://platform.openai.com/api-keys
   - Create a new API key
   - Add to `.env` file: `OPENAI_API_KEY=your_key_here`

2. **Perplexity API Key:**
   - Visit: https://www.perplexity.ai/settings/api
   - Generate API key
   - Add to `.env` file: `PERPLEXITY_API_KEY=your_key_here`

### Environment File Example

Create a `.env` file in the project root:

```env
# API Keys (at least one required)
OPENAI_API_KEY=sk-your-openai-key-here
PERPLEXITY_API_KEY=pplx-your-perplexity-key-here

# Optional: Google Sheets (for data storage)
GOOGLE_SPREADSHEET_ID=your-spreadsheet-id-here

# Application Settings
LOG_LEVEL=INFO
DEBUG_MODE=false
```

## 🌐 Accessing the Application

Once started, the application will be available at:
- **Local:** http://localhost:8501
- **Network:** http://your-ip-address:8501

## 🔧 Troubleshooting

### Common Issues

1. **"Python not found"**
   - Install Python 3.8+ from https://python.org
   - Add Python to PATH during installation

2. **"Port 8501 already in use"**
   - Kill existing process: `netstat -ano | findstr :8501`
   - Or use different port: `streamlit run streamlit_app.py --server.port=8502`

3. **"Module not found"**
   - Ensure virtual environment is activated
   - Reinstall dependencies: `pip install -r requirements.txt`

4. **"API key error"**
   - Check `.env` file exists and has correct API keys
   - Verify API keys are valid and have sufficient credits

5. **"Docker connection error"**
   - Ensure Docker Desktop is running
   - Restart Docker Desktop if needed

### Getting Help

1. **Check logs** in the `logs/` directory
2. **Run tests:** `python test_sample.py`
3. **Test connections:** Use the "System Status" page in the web app

## 📱 Using the Application

### Dashboard
- View system status and quick monitoring options
- Run sample queries with one click

### Brand Monitoring
- Enter custom search queries
- Monitor brand mentions in real-time
- View detailed results and agent performance

### Historical Analysis
- View past monitoring results
- Export data in CSV/JSON format
- Analyze trends over time

### Settings
- Configure brand detection parameters
- Adjust workflow settings
- Modify LLM configurations

### System Status
- Test API connections
- View performance metrics
- Check system health

## 🚀 Next Steps

1. **Configure your target brand** in the Settings page
2. **Set up Google Sheets** for data storage (optional)
3. **Run your first monitoring** with sample queries
4. **Explore advanced features** like historical analysis

---

**Need help?** Check the main README.md or STREAMLIT_DEPLOYMENT.md for detailed documentation. 