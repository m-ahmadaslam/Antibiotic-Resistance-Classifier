"""
Predictions Page Module
======================

Contains the predictions functionality for the healthcare application.
"""

import streamlit as st
import pandas as pd
import os
import plotly.express as px

def predictions_page(pkl_manager):
    """Page for making predictions on new data"""
    st.header("🔮 Make Predictions")
    
    # Check if models exist
    model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
    
    if not model_files:
        st.warning("⚠️ No trained models found. Please train a model first.")
        if st.button("Go to Training Page"):
            st.session_state.page = "📁 Upload & Train"
            st.rerun()
        return
    
    # Model selection
    st.subheader("🤖 Select Model")
    selected_model = st.selectbox(
        "Choose a model:",
        model_files,
        format_func=lambda x: f"{x.split('_')[2]} - {x.split('_')[1]} ({x.split('_')[3].split('.')[0]})"
    )
    
    # Data source selection
    st.subheader("📊 Choose Data Source")
    data_source = st.radio(
        "Select data source for predictions:",
        ["Use existing training data", "Upload new data file"],
        help="Choose whether to use the same data you trained with or upload new data"
    )
    
    if data_source == "Use existing training data":
        # Show available data files
        data_files = [f for f in os.listdir('data') if f.endswith(('.xlsx', '.xls'))]
        
        if not data_files:
            st.warning("⚠️ No data files found. Please upload data first.")
            if st.button("Go to Training Page"):
                st.session_state.page = "📁 Upload & Train"
                st.rerun()
            return
        
        selected_data_file = st.selectbox(
            "Choose a data file:",
            data_files,
            help="Select the Excel file you want to make predictions on"
        )
        
        if selected_data_file and selected_model:
            pred_file_path = os.path.join('data', selected_data_file)
            st.success(f"✅ Using existing data: {selected_data_file}")
            
            # Make predictions button
            if st.button("🔮 Make Predictions", type="primary", use_container_width=True):
                with st.spinner("Making predictions..."):
                    try:
                        # Load model and make predictions
                        model_path = os.path.join('models', selected_model)
                        results = pkl_manager.predict_with_pkl_model(
                            model_path=model_path,
                            new_data_path=pred_file_path,
                            output_path=f"predictions/predictions_{selected_data_file.replace('.xls', '.xlsx')}"
                        )
                        
                        if results is not None:
                            st.success("✅ Predictions completed successfully!")
                            
                            # Show results summary
                            st.subheader("📊 Prediction Summary")
                            
                            total_samples = len(results)
                            matches = sum(results['Predicted_Result'])
                            mismatches = total_samples - matches
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Samples", total_samples)
                            with col2:
                                st.metric("Predicted Matches", matches)
                            with col3:
                                st.metric("Predicted Mismatches", mismatches)
                            
                            # Show sample predictions
                            st.subheader("📋 Sample Predictions")
                            display_cols = ['EXAM_RESULT', 'ANTIBIOTIC_ORDER_NAME', 'Predicted_Class', 'Prediction_Probability']
                            available_cols = [col for col in display_cols if col in results.columns]
                            st.dataframe(results[available_cols].head(10), use_container_width=True)
                            
                            # Download results
                            csv = results.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions (CSV)",
                                data=csv,
                                file_name=f"predictions_{selected_data_file.replace('.xlsx', '.csv').replace('.xls', '.csv')}",
                                mime="text/csv"
                            )
                            
                            # Show probability distribution
                            if 'Prediction_Probability' in results.columns:
                                st.subheader("📈 Prediction Probability Distribution")
                                fig = px.histogram(
                                    results, 
                                    x='Prediction_Probability',
                                    nbins=20,
                                    title="Distribution of Prediction Probabilities"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("❌ No results returned from prediction")
                        
                    except Exception as e:
                        import traceback
                        st.error(f"❌ Error making predictions: {str(e)}")
                        st.error(f"Full error: {traceback.format_exc()}")
    
    else:
        # File upload for predictions
        st.subheader("📤 Upload New Data for Prediction")
        
        prediction_file = st.file_uploader(
            "Choose an Excel file for prediction",
            type=['xlsx', 'xls'],
            help="Upload new data to make predictions"
        )
        
        if prediction_file and selected_model:
            # Save uploaded file
            pred_file_path = os.path.join('data', prediction_file.name)
            with open(pred_file_path, 'wb') as f:
                f.write(prediction_file.getbuffer())
            
            st.success(f"✅ File uploaded: {prediction_file.name}")
            
            # Make predictions button
            if st.button("🔮 Make Predictions", type="primary", use_container_width=True):
                with st.spinner("Making predictions..."):
                    try:
                        # Load model and make predictions
                        model_path = os.path.join('models', selected_model)
                        results = pkl_manager.predict_with_pkl_model(
                            model_path=model_path,
                            new_data_path=pred_file_path,
                            output_path=f"predictions/predictions_{prediction_file.name.replace('.xls', '.xlsx')}"
                        )
                        
                        if results is not None:
                            st.success("✅ Predictions completed successfully!")
                            
                            # Show results summary
                            st.subheader("📊 Prediction Summary")
                            
                            total_samples = len(results)
                            matches = sum(results['Predicted_Result'])
                            mismatches = total_samples - matches
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Samples", total_samples)
                            with col2:
                                st.metric("Predicted Matches", matches)
                            with col3:
                                st.metric("Predicted Mismatches", mismatches)
                            
                            # Show sample predictions
                            st.subheader("📋 Sample Predictions")
                            display_cols = ['EXAM_RESULT', 'ANTIBIOTIC_ORDER_NAME', 'Predicted_Class', 'Prediction_Probability']
                            available_cols = [col for col in display_cols if col in results.columns]
                            st.dataframe(results[available_cols].head(10), use_container_width=True)
                            
                            # Download results
                            csv = results.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions (CSV)",
                                data=csv,
                                file_name=f"predictions_{prediction_file.name.replace('.xlsx', '.csv')}",
                                mime="text/csv"
                            )
                            
                            # Show probability distribution
                            if 'Prediction_Probability' in results.columns:
                                st.subheader("📈 Prediction Probability Distribution")
                                fig = px.histogram(
                                    results, 
                                    x='Prediction_Probability',
                                    nbins=20,
                                    title="Distribution of Prediction Probabilities"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("❌ No results returned from prediction")
                        
                    except Exception as e:
                        import traceback
                        st.error(f"❌ Error making predictions: {str(e)}")
                        st.error(f"Full error: {traceback.format_exc()}")
