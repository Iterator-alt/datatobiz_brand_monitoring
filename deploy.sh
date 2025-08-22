#!/bin/bash

# DataTobiz Brand Monitoring System - Deployment Script
# This script helps deploy the Streamlit application

set -e

echo "🚀 DataTobiz Brand Monitoring System - Deployment Script"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required files exist
check_requirements() {
    print_status "Checking requirements..."
    
    required_files=("config.yaml" "requirements.txt" "streamlit_app.py")
    missing_files=()
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            missing_files+=("$file")
        fi
    done
    
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        print_error "Missing required files: ${missing_files[*]}"
        exit 1
    fi
    
    print_success "All required files found"
}

# Check environment variables
check_environment() {
    print_status "Checking environment variables..."
    
    required_vars=("OPENAI_API_KEY" "PERPLEXITY_API_KEY")
    missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        print_warning "Missing environment variables: ${missing_vars[*]}"
        print_warning "Please set these variables or create a .env file"
    else
        print_success "Environment variables configured"
    fi
}

# Install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    if [[ -d "venv" ]]; then
        print_status "Using existing virtual environment"
        source venv/bin/activate
    else
        print_status "Creating virtual environment..."
        python -m venv venv
        source venv/bin/activate
    fi
    
    pip install --upgrade pip
    pip install -r requirements.txt
    
    print_success "Dependencies installed"
}

# Run tests
run_tests() {
    print_status "Running system tests..."
    
    if python test_sample.py; then
        print_success "Tests passed"
    else
        print_warning "Some tests failed, but continuing with deployment"
    fi
}

# Deploy with Docker
deploy_docker() {
    print_status "Deploying with Docker..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Build and run with docker-compose
    docker-compose up --build -d
    
    print_success "Docker deployment completed"
    print_status "Application is running at: http://localhost:8501"
}

# Deploy locally
deploy_local() {
    print_status "Deploying locally..."
    
    # Create logs directory if it doesn't exist
    mkdir -p logs
    
    # Run Streamlit
    print_status "Starting Streamlit application..."
    print_status "Application will be available at: http://localhost:8501"
    
    streamlit run streamlit_app.py --server.port=8501
}

# Main deployment function
main() {
    local deployment_type=${1:-local}
    
    case $deployment_type in
        "docker")
            check_requirements
            check_environment
            deploy_docker
            ;;
        "local")
            check_requirements
            check_environment
            install_dependencies
            run_tests
            deploy_local
            ;;
        "test")
            check_requirements
            check_environment
            install_dependencies
            run_tests
            ;;
        *)
            echo "Usage: $0 {local|docker|test}"
            echo ""
            echo "Deployment options:"
            echo "  local   - Deploy locally with Streamlit"
            echo "  docker  - Deploy with Docker Compose"
            echo "  test    - Run tests only"
            exit 1
            ;;
    esac
}

# Run main function with arguments
main "$@" 