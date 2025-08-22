# 🚀 Streamlit Cloud Deployment Guide

## Quick Deploy to Streamlit Cloud

### Prerequisites
1. **GitHub Account** - Create one at https://github.com
2. **Streamlit Cloud Account** - Sign up at https://share.streamlit.io
3. **API Keys** - OpenAI and/or Perplexity API keys

### Step 1: Push to GitHub

1. **Create a new repository on GitHub:**
   - Go to https://github.com/new
   - Name it: `datatobiz-brand-monitoring`
   - Make it **Public** (required for free Streamlit Cloud)
   - Don't initialize with README

2. **Push your code to GitHub:**
   ```bash
   git add .
   git commit -m "Initial commit: DataTobiz Brand Monitoring System"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/datatobiz-brand-monitoring.git
   git push -u origin main
   ```

### Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud:**
   - Visit https://share.streamlit.io
   - Sign in with GitHub

2. **Deploy your app:**
   - Click "New app"
   - Select your repository: `datatobiz-brand-monitoring`
   - Set main file path: `streamlit_app.py`
   - Click "Deploy!"

### Step 3: Configure Secrets

1. **In Streamlit Cloud dashboard:**
   - Go to your app settings
   - Click "Secrets"
   - Add your API keys:

   ```toml
   OPENAI_API_KEY = "sk-your-actual-openai-key"
   PERPLEXITY_API_KEY = "pplx-your-actual-perplexity-key"
   GOOGLE_SPREADSHEET_ID = "your-spreadsheet-id"
   ```

2. **Save and redeploy**

### Step 4: Access Your App

Your app will be available at:
`https://your-app-name-your-username.streamlit.app`

## 🔧 Troubleshooting

### Common Issues

1. **"Module not found" errors:**
   - Check `requirements.txt` has all dependencies
   - Ensure all imports are correct

2. **API key errors:**
   - Verify secrets are set correctly in Streamlit Cloud
   - Check API keys are valid and have credits

3. **Google Sheets errors:**
   - Ensure credentials.json is properly configured
   - Check spreadsheet permissions

### Local Testing Before Deployment

```bash
# Test locally first
streamlit run streamlit_app.py --server.port=8501
```

## 📝 Important Notes

- **Never commit API keys** to Git
- **Use Streamlit Cloud secrets** for sensitive data
- **Keep repository public** for free tier
- **Monitor usage** to avoid API costs

## 🎯 Next Steps

1. **Customize the app** for your needs
2. **Add more features** like email notifications
3. **Set up monitoring** and alerts
4. **Scale up** if needed

---

**Need help?** Check the main README.md for detailed documentation. 