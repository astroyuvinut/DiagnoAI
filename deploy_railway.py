# Railway.app deployment script
# This file helps deploy to Railway.app for free web hosting

import os
import subprocess
import sys

def create_railway_files():
    """Create files needed for Railway deployment"""
    
    # Procfile for Railway
    with open('Procfile', 'w') as f:
        f.write('web: python -m app.flask_app\n')
    
    # Railway.json configuration
    railway_config = {
        "build": {
            "builder": "NIXPACKS"
        },
        "deploy": {
            "startCommand": "python -m app.flask_app",
            "healthcheckPath": "/health",
            "healthcheckTimeout": 100,
            "restartPolicyType": "ON_FAILURE",
            "restartPolicyMaxRetries": 10
        }
    }
    
    import json
    with open('railway.json', 'w') as f:
        json.dump(railway_config, f, indent=2)
    
    # Runtime.txt for Python version
    with open('runtime.txt', 'w') as f:
        f.write('python-3.11.9\n')  # Railway supports 3.11 better than 3.13
    
    print("✅ Railway deployment files created!")
    print("Files created: Procfile, railway.json, runtime.txt")

def create_heroku_files():
    """Create files needed for Heroku deployment"""
    
    # Procfile for Heroku
    with open('Procfile', 'w') as f:
        f.write('web: python -m app.flask_app\n')
    
    # Runtime.txt for Python version
    with open('runtime.txt', 'w') as f:
        f.write('python-3.11.9\n')
    
    print("✅ Heroku deployment files created!")
    print("Files created: Procfile, runtime.txt")

def create_render_files():
    """Create files needed for Render deployment"""
    
    # render.yaml for Render
    render_config = """services:
  - type: web
    name: diagnoai
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python -m app.flask_app
    healthCheckPath: /health
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.9
"""
    
    with open('render.yaml', 'w') as f:
        f.write(render_config)
    
    print("✅ Render deployment files created!")
    print("Files created: render.yaml")

if __name__ == "__main__":
    print("🚀 Creating deployment files for multiple platforms...")
    print()
    
    create_railway_files()
    print()
    create_heroku_files()
    print()
    create_render_files()
    print()
    
    print("📋 Next steps:")
    print("1. Choose a platform (Railway, Heroku, or Render)")
    print("2. Create account and connect GitHub repo")
    print("3. Deploy!")
    print()
    print("🌐 Your app is currently running locally at: http://localhost:5000")
