"""
Model Management Page Module
============================

Contains the model management functionality for the healthcare application.
"""

import streamlit as st
import pandas as pd
import os
import pickle

def model_management_page():
    """Page for managing saved models"""
    st.header("📊 Model Management")
    
    # List saved models
    st.subheader("💾 Saved Models")
    
    model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
    
    if not model_files:
        st.warning("No saved models found.")
        return
    
    # Display models in a table
    model_data = []
    for model_file in model_files:
        file_path = os.path.join('models', model_file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        
        try:
            with open(file_path, 'rb') as f:
                model_package = pickle.load(f)
                model_type = model_package.get('model_type', 'Unknown')
                timestamp = model_package.get('timestamp', 'Unknown')
                accuracy = model_package.get('metadata', {}).get('accuracy', 'N/A')
        except:
            model_type = 'Unknown'
            timestamp = 'Unknown'
            accuracy = 'N/A'
        
        model_data.append({
            'Model File': model_file,
            'Type': model_type,
            'Accuracy': f"{accuracy:.4f}" if isinstance(accuracy, float) else accuracy,
            'Size (MB)': f"{file_size:.2f}",
            'Created': timestamp
        })
    
    if model_data:
        df = pd.DataFrame(model_data)
        st.dataframe(df, use_container_width=True)
        
        # Model actions
        st.subheader("🔧 Model Actions")
        
        selected_model = st.selectbox(
            "Select a model for actions:",
            model_files
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 View Details", use_container_width=True):
                show_model_details(selected_model)
        
        with col2:
            if st.button("🗑️ Delete Model", use_container_width=True):
                if st.checkbox("I confirm I want to delete this model"):
                    try:
                        os.remove(os.path.join('models', selected_model))
                        st.success("Model deleted successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting model: {str(e)}")
        
        with col3:
            if st.button("📥 Download Model", use_container_width=True):
                model_path = os.path.join('models', selected_model)
                with open(model_path, 'rb') as f:
                    st.download_button(
                        label="Download PKL File",
                        data=f.read(),
                        file_name=selected_model,
                        mime="application/octet-stream"
                    )

def show_model_details(model_file):
    """Show detailed information about a specific model"""
    model_path = os.path.join('models', model_file)
    
    try:
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
        
        st.subheader(f"📊 Model Details: {model_file}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Information:**")
            st.write(f"- Model Type: {model_package.get('model_type', 'Unknown')}")
            st.write(f"- Created: {model_package.get('timestamp', 'Unknown')}")
            st.write(f"- Version: {model_package.get('version', 'Unknown')}")
            
            if 'feature_columns' in model_package and model_package['feature_columns']:
                st.write(f"- Features: {len(model_package['feature_columns'])}")
        
        with col2:
            st.write("**Performance Metrics:**")
            metadata = model_package.get('metadata', {})
            if 'accuracy' in metadata:
                st.write(f"- Accuracy: {metadata['accuracy']:.4f}")
            if 'f1_score' in metadata:
                st.write(f"- F1-Score: {metadata['f1_score']:.4f}")
            if 'training_samples' in metadata:
                st.write(f"- Training Samples: {metadata['training_samples']}")
            if 'test_samples' in metadata:
                st.write(f"- Test Samples: {metadata['test_samples']}")
        
        # Show feature importance if available
        if 'feature_columns' in model_package and model_package['feature_columns']:
            st.subheader("🎯 Feature Information")
            feature_cols = model_package['feature_columns']
            
            # Count different types of features
            organism_features = [f for f in feature_cols if f.startswith('organism_')]
            antibiotic_features = [f for f in feature_cols if f.startswith('antibiotic_')]
            other_features = [f for f in feature_cols if not f.startswith(('organism_', 'antibiotic_'))]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Organism Features", len(organism_features))
            with col2:
                st.metric("Antibiotic Features", len(antibiotic_features))
            with col3:
                st.metric("Other Features", len(other_features))
            
            # Show sample features
            with st.expander("View Feature List"):
                st.write("**Organism Features:**")
                st.write(organism_features[:10])  # Show first 10
                
                st.write("**Antibiotic Features:**")
                st.write(antibiotic_features[:10])  # Show first 10
                
                st.write("**Other Features:**")
                st.write(other_features[:10])  # Show first 10
    
    except Exception as e:
        st.error(f"Error loading model details: {str(e)}")
