"""
Visualizations Page Module
==========================

Contains the visualizations functionality for the healthcare application.
"""

import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go

def visualizations_page():
    """Page for displaying model visualizations"""
    st.header("📈 Model Visualizations")
    
    # Check if we have training results
    if st.session_state.training_results is None:
        st.warning("⚠️ No training results available. Please train a model first.")
        if st.button("Go to Training Page"):
            st.session_state.page = "📁 Upload & Train"
            st.rerun()
        return
    
    # Check if visualization files exist
    confusion_matrix_path = "results/confusion_matrix.png"
    feature_importance_path = "results/feature_importance.png"
    model_comparison_path = "results/model_comparison.png"
    
    # Confusion Matrix
    st.subheader("🎯 Confusion Matrix")
    if os.path.exists(confusion_matrix_path):
        st.image(confusion_matrix_path, caption="Confusion Matrix", use_column_width=True)
    else:
        st.info("Confusion matrix will be generated after training.")
    
    # Feature Importance
    st.subheader("📊 Feature Importance")
    if os.path.exists(feature_importance_path):
        st.image(feature_importance_path, caption="Feature Importance", use_column_width=True)
    else:
        st.info("Feature importance will be generated after training.")
    
    # Model Comparison
    st.subheader("📈 Model Comparison")
    if os.path.exists(model_comparison_path):
        st.image(model_comparison_path, caption="Model Performance Comparison", use_column_width=True)
    else:
        st.info("Model comparison will be generated after training.")
    
    # Interactive visualizations
    if st.session_state.model_metrics:
        st.subheader("📊 Interactive Model Metrics")
        
        # Create interactive comparison chart
        models = list(st.session_state.model_metrics.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Prepare data for plotting
        plot_data = []
        for model in models:
            for metric in metrics:
                plot_data.append({
                    'Model': model,
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': st.session_state.model_metrics[model][metric]
                })
        
        df_plot = pd.DataFrame(plot_data)
        
        # Create interactive bar chart
        fig = px.bar(
            df_plot, 
            x='Model', 
            y='Value', 
            color='Metric',
            title="Model Performance Comparison",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Create radar chart for best model
        if st.session_state.best_model_name:
            best_model_metrics = st.session_state.model_metrics[st.session_state.best_model_name]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[best_model_metrics['accuracy'], best_model_metrics['precision'], 
                   best_model_metrics['recall'], best_model_metrics['f1_score']],
                theta=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                fill='toself',
                name=st.session_state.best_model_name
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title=f"Performance Radar Chart - {st.session_state.best_model_name}"
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
