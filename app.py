"""
Healthcare Antibiotic Classification - Refactored Streamlit App
==============================================================

Clean, modular UI application with file upload, model training, predictions, and chatbot.
"""

import streamlit as st
import os

# Import our custom modules
from data_processor import HealthcareDataProcessor
from model_trainer import HealthcareModelTrainer
from model_pkl_manager import HealthcareModelPKLManager

# Import page modules
from modules.dashboard import dashboard_page
from modules.upload_train import upload_train_page
from modules.predictions import predictions_page
from modules.visualizations import visualizations_page
from modules.chatbot import chatbot_page
from modules.model_management import model_management_page

# Page configuration
st.set_page_config(
    page_title="Healthcare Antibiotic Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide the automatic page navigation
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    .stApp > header {display: none;}
    .stApp > footer {display: none;}
    .stApp > .main > .block-container {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
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
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

class HealthcareApp:
    def __init__(self):
        self.data_processor = HealthcareDataProcessor()
        self.model_trainer = HealthcareModelTrainer()
        self.pkl_manager = HealthcareModelPKLManager()
        
        # Create directories if they don't exist
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('predictions', exist_ok=True)
        
        # Initialize session state for persistence
        if 'training_results' not in st.session_state:
            st.session_state.training_results = None
        if 'best_model_name' not in st.session_state:
            st.session_state.best_model_name = None
        if 'model_metrics' not in st.session_state:
            st.session_state.model_metrics = None
        if 'uploaded_file_name' not in st.session_state:
            st.session_state.uploaded_file_name = None
        if 'file_data' not in st.session_state:
            st.session_state.file_data = None
        if 'training_completed' not in st.session_state:
            st.session_state.training_completed = False
    
    def main_app(self):
        """Main application interface"""
        
        # Header
        st.markdown('<h1 class="main-header">🏥 Healthcare Antibiotic Classifier</h1>', unsafe_allow_html=True)
        
        # Initialize session state
        if 'page' not in st.session_state:
            st.session_state.page = "🏠 Dashboard"
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        
        # Use radio buttons instead of selectbox for better UX
        page = st.sidebar.radio(
            "Choose a page:",
            ["🏠 Dashboard", "📁 Upload & Train", "🔮 Make Predictions", "📈 Visualizations", "🤖 AI Chatbot", "📊 Model Management"],
            index=["🏠 Dashboard", "📁 Upload & Train", "🔮 Make Predictions", "📈 Visualizations", "🤖 AI Chatbot", "📊 Model Management"].index(st.session_state.page)
        )
        
        # Update session state when page changes
        if page != st.session_state.page:
            st.session_state.page = page
            st.rerun()
        
        # Route to appropriate page
        if page == "🏠 Dashboard":
            dashboard_page()
        elif page == "📁 Upload & Train":
            upload_train_page(self.data_processor, self.model_trainer, self.pkl_manager)
        elif page == "🔮 Make Predictions":
            predictions_page(self.pkl_manager)
        elif page == "📈 Visualizations":
            visualizations_page()
        elif page == "🤖 AI Chatbot":
            chatbot_page()
        elif page == "📊 Model Management":
            model_management_page()

def main():
    """Main function to run the Streamlit app"""
    app = HealthcareApp()
    app.main_app()

if __name__ == "__main__":
    main()
