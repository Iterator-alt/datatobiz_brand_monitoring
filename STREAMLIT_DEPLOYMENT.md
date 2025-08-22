# DataTobiz Brand Monitoring System - Streamlit Deployment Guide

This guide provides comprehensive instructions for deploying the DataTobiz Brand Monitoring System as a Streamlit web application.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git
- API keys for OpenAI and/or Perplexity
- Google Cloud credentials (for Google Sheets integration)

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd datatobiz-brand-monitoring

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
PERPLEXITY_API_KEY=your_perplexity_api_key_here
GOOGLE_SPREADSHEET_ID=your_google_spreadsheet_id_here

# Optional settings
LOG_LEVEL=INFO
DEBUG_MODE=false
```

### 3. Run the Application

```bash
# Start Streamlit application
streamlit run streamlit_app.py
```

The application will be available at: http://localhost:8501

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

1. **Build and run with docker-compose:**

```bash
# Build and start the application
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

2. **Access the application:**
   - URL: http://localhost:8501
   - The application will automatically restart if it crashes

### Using Docker directly

```bash
# Build the Docker image
docker build -t datatobiz-brand-monitoring .

# Run the container
docker run -d \
  --name datatobiz-brand-monitoring \
  -p 8501:8501 \
  -e OPENAI_API_KEY=your_key \
  -e PERPLEXITY_API_KEY=your_key \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/credentials.json:/app/credentials.json:ro \
  datatobiz-brand-monitoring
```

## ☁️ Cloud Deployment

### Heroku Deployment

1. **Create Heroku app:**

```bash
# Install Heroku CLI
# Create new app
heroku create your-app-name

# Set environment variables
heroku config:set OPENAI_API_KEY=your_key
heroku config:set PERPLEXITY_API_KEY=your_key
heroku config:set GOOGLE_SPREADSHEET_ID=your_id

# Deploy
git push heroku main
```

2. **Create `Procfile` for Heroku:**

```
web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

### AWS Deployment

1. **Using AWS ECS:**

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com

docker build -t datatobiz-brand-monitoring .
docker tag datatobiz-brand-monitoring:latest your-account.dkr.ecr.us-east-1.amazonaws.com/datatobiz-brand-monitoring:latest
docker push your-account.dkr.ecr.us-east-1.amazonaws.com/datatobiz-brand-monitoring:latest
```

2. **Create ECS task definition and service**

### Google Cloud Platform

1. **Using Cloud Run:**

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/your-project/datatobiz-brand-monitoring
gcloud run deploy datatobiz-brand-monitoring \
  --image gcr.io/your-project/datatobiz-brand-monitoring \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=your_key,PERPLEXITY_API_KEY=your_key
```

## 🔧 Configuration

### Streamlit Configuration

The application uses two configuration files:

1. **`.streamlit/config.toml`** - Development settings
2. **`streamlit_config.toml`** - Production settings

Key configuration options:

```toml
[server]
port = 8501
address = "0.0.0.0"  # For production
headless = true      # For production

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Yes* |
| `PERPLEXITY_API_KEY` | Perplexity API key | Yes* |
| `GOOGLE_SPREADSHEET_ID` | Google Sheets ID | No |
| `LOG_LEVEL` | Logging level | No |
| `DEBUG_MODE` | Debug mode | No |

*At least one LLM API key is required

## 📊 Application Features

### Dashboard
- System status overview
- Quick monitoring interface
- Recent results display

### Brand Monitoring
- Multi-query input
- Real-time monitoring
- Agent performance tracking

### Historical Analysis
- Data visualization with Plotly
- Export capabilities (CSV/JSON)
- Trend analysis

### Settings
- Brand configuration
- Workflow settings
- LLM model configuration

### System Status
- Connection testing
- Performance metrics
- Configuration overview

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors:**
   ```bash
   # Ensure you're in the correct directory
   cd datatobiz-brand-monitoring
   
   # Check Python path
   python -c "import sys; print(sys.path)"
   ```

2. **API Key Issues:**
   ```bash
   # Verify environment variables
   echo $OPENAI_API_KEY
   echo $PERPLEXITY_API_KEY
   ```

3. **Port Already in Use:**
   ```bash
   # Kill existing process
   lsof -ti:8501 | xargs kill -9
   
   # Or use different port
   streamlit run streamlit_app.py --server.port=8502
   ```

4. **Docker Issues:**
   ```bash
   # Clean up Docker
   docker system prune -a
   docker volume prune
   
   # Rebuild without cache
   docker-compose build --no-cache
   ```

### Logs and Debugging

1. **View Streamlit logs:**
   ```bash
   # Local deployment
   streamlit run streamlit_app.py --logger.level=debug
   
   # Docker deployment
   docker-compose logs -f
   ```

2. **Check application logs:**
   ```bash
   # View log files
   tail -f logs/brand_monitoring.log
   ```

## 🔒 Security Considerations

### Production Security

1. **Environment Variables:**
   - Never commit API keys to version control
   - Use secure environment variable management
   - Rotate API keys regularly

2. **Network Security:**
   - Use HTTPS in production
   - Configure firewall rules
   - Enable authentication if needed

3. **Container Security:**
   - Run containers as non-root user
   - Use minimal base images
   - Regularly update dependencies

### Authentication (Optional)

To add authentication to the Streamlit app:

```python
# Add to streamlit_app.py
import streamlit_authenticator as stauth

# Configure authentication
names = ['admin']
usernames = ['admin']
passwords = ['password123']

hashed_passwords = stauth.Hasher(passwords).generate()

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
    'some_cookie_name', 'some_key', cookie_expiry_days=30)

name, authentication_status, username = authenticator.login('Login', 'main')
```

## 📈 Monitoring and Scaling

### Health Checks

The application includes health check endpoints:

- **Streamlit Health:** `http://localhost:8501/_stcore/health`
- **Application Health:** Custom health check in the app

### Scaling Options

1. **Horizontal Scaling:**
   - Use load balancer with multiple instances
   - Configure session state management
   - Use external database for state

2. **Vertical Scaling:**
   - Increase container resources
   - Optimize memory usage
   - Use caching strategies

### Performance Optimization

1. **Caching:**
   ```python
   @st.cache_data
   def expensive_function():
       # Cached function
       pass
   ```

2. **Async Operations:**
   - Use async/await for API calls
   - Implement background tasks
   - Use connection pooling

## 🚀 Advanced Deployment

### Kubernetes Deployment

1. **Create deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: datatobiz-brand-monitoring
spec:
  replicas: 3
  selector:
    matchLabels:
      app: datatobiz-brand-monitoring
  template:
    metadata:
      labels:
        app: datatobiz-brand-monitoring
    spec:
      containers:
      - name: app
        image: datatobiz-brand-monitoring:latest
        ports:
        - containerPort: 8501
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-key
```

2. **Create service.yaml:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: datatobiz-brand-monitoring-service
spec:
  selector:
    app: datatobiz-brand-monitoring
  ports:
  - port: 80
    targetPort: 8501
  type: LoadBalancer
```

### CI/CD Pipeline

Example GitHub Actions workflow:

```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Build Docker image
      run: docker build -t datatobiz-brand-monitoring .
    
    - name: Deploy to Cloud Run
      run: |
        gcloud auth configure-docker
        docker tag datatobiz-brand-monitoring gcr.io/${{ secrets.GCP_PROJECT }}/datatobiz-brand-monitoring
        docker push gcr.io/${{ secrets.GCP_PROJECT }}/datatobiz-brand-monitoring
        gcloud run deploy datatobiz-brand-monitoring \
          --image gcr.io/${{ secrets.GCP_PROJECT }}/datatobiz-brand-monitoring \
          --platform managed \
          --region us-central1 \
          --allow-unauthenticated
```

## 📞 Support

For deployment issues:

1. **Check the logs:** `docker-compose logs -f`
2. **Verify configuration:** Ensure all environment variables are set
3. **Test locally first:** Run `python test_sample.py`
4. **Check system requirements:** Ensure sufficient memory and CPU

## 🔄 Updates and Maintenance

### Updating the Application

1. **Pull latest changes:**
   ```bash
   git pull origin main
   ```

2. **Update dependencies:**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

3. **Rebuild Docker image:**
   ```bash
   docker-compose build --no-cache
   docker-compose up -d
   ```

### Backup and Recovery

1. **Backup configuration:**
   ```bash
   cp config.yaml config.yaml.backup
   cp .env .env.backup
   ```

2. **Backup data:**
   ```bash
   # Export Google Sheets data
   # Backup logs
   tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/
   ```

---

**Happy Deploying! 🚀**

For more information, refer to the main README.md file or contact the development team. 