from flask import Flask
from flask_cors import CORS
from app.routes import routes
from app.vector_store import init_vectorstore
import os

def create_app():
    app = Flask(__name__, 
                template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))
    
    # Get environment
    is_production = os.getenv('FLASK_ENV') == 'production'
    
    # Configure CORS based on environment
    if is_production:
        # Production CORS settings
        CORS(app, resources={
            r"/*": {
                "origins": [
                    "https://*.netlify.app",  # All Netlify domains
                    "https://your-site-name.netlify.app"  # Your specific Netlify domain
                ],
                "methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["Content-Type"]
            }
        })
    else:
        # Development CORS settings - more permissive
        CORS(app, resources={r"/*": {"origins": "*"}})
    
    # Initialize vector store during app startup
    print("Starting application initialization...")
    try:
        init_vectorstore()
        print("Vector store initialization complete!")
    except Exception as e:
        print(f"Error initializing vector store: {str(e)}")
    
    app.register_blueprint(routes)
    
    # Print the port being used
    port = int(os.getenv("PORT", 5001))
    print(f"Application will run on port: {port}")
    
    return app
