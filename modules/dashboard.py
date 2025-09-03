"""
Dashboard Page Module
====================

Contains the dashboard functionality for the healthcare application.
"""

import streamlit as st
import os

def dashboard_page():
    """Dashboard page showing overview and statistics"""
    st.header("📊 Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Get statistics
    model_count = len([f for f in os.listdir('models') if f.endswith('.pkl')])
    data_files = len([f for f in os.listdir('data') if f.endswith(('.xlsx', '.xls'))])
    prediction_files = len([f for f in os.listdir('predictions') if f.endswith('.xlsx')])
    
    with col1:
        st.metric("Trained Models", model_count)
    with col2:
        st.metric("Data Files", data_files)
    with col3:
        st.metric("Predictions Made", prediction_files)
    with col4:
        st.metric("System Status", "🟢 Active")
    
    # Recent activity
    st.subheader("Recent Activity")
    
    # Show recent models
    if model_count > 0:
        model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
        model_files.sort(reverse=True)
        
        st.write("**Recent Models:**")
        for i, model_file in enumerate(model_files[:3]):
            st.write(f"• {model_file}")
    
    # Show recent training results if available
    if st.session_state.training_results and st.session_state.best_model_name:
        st.subheader("🏆 Latest Training Results")
        
        best_results = st.session_state.training_results[st.session_state.best_model_name]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Best Model", st.session_state.best_model_name)
        with col2:
            st.metric("Accuracy", f"{best_results['accuracy']:.4f}")
        with col3:
            st.metric("F1-Score", f"{best_results['f1_score']:.4f}")
        with col4:
            st.metric("Status", "✅ Trained")
    
    # Quick actions
    st.subheader("Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚀 Train New Model", use_container_width=True, key="nav_train"):
            st.session_state.page = "📁 Upload & Train"
            st.rerun()
    
    with col2:
        if st.button("🔮 Make Predictions", use_container_width=True, key="nav_predict"):
            st.session_state.page = "🔮 Make Predictions"
            st.rerun()
    
    with col3:
        if st.button("📈 View Visualizations", use_container_width=True, key="nav_viz"):
            st.session_state.page = "📈 Visualizations"
            st.rerun()
    
    with col4:
        if st.button("🤖 Chat with AI", use_container_width=True, key="nav_chat"):
            st.session_state.page = "🤖 AI Chatbot"
            st.rerun()
