"""
Upload & Train Page Module
=========================

Contains the upload and training functionality for the healthcare application.
"""

import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime

def upload_train_page(data_processor, model_trainer, pkl_manager):
    """Page for uploading data and training models"""
    st.header("📁 Upload & Train Model")
    
    # File upload section
    st.subheader("📤 Upload Your Data")
    
    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=['xlsx', 'xls'],
        help="Upload your healthcare data in Excel format"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        file_path = os.path.join('data', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Save file info in session state
        st.session_state.uploaded_file_name = uploaded_file.name
        
        st.success(f"✅ File uploaded successfully: {uploaded_file.name}")
        
        # Show file info
        try:
            # Try different engines for Excel files
            engines_to_try = ['openpyxl', 'xlrd', None]
            
            for engine in engines_to_try:
                try:
                    if engine:
                        data = pd.read_excel(file_path, engine=engine)
                    else:
                        data = pd.read_excel(file_path)
                    print(f"Successfully loaded with engine: {engine}")
                    break
                except Exception as e:
                    print(f"Failed with engine {engine}: {str(e)}")
                    continue
            else:
                raise Exception("Could not read Excel file with any available engine")
            
            st.session_state.file_data = data
            st.write(f"**File Info:** {data.shape[0]} rows, {data.shape[1]} columns")
            
            # Show sample data
            st.subheader("📋 Sample Data")
            st.dataframe(data.head(), use_container_width=True)
            
            # Training options
            st.subheader("⚙️ Training Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
                random_state = st.number_input("Random State", 1, 100, 42)
            
            with col2:
                models_to_train = st.multiselect(
                    "Models to Train",
                    ["XGBoost", "CatBoost", "Random Forest", "Logistic Regression"],
                    default=["XGBoost", "CatBoost"]
                )
            
            # Train button
            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                with st.spinner("Training models... This may take several minutes."):
                    try:
                        # Load and preprocess data
                        data = data_processor.load_and_explore_data(file_path)
                        processed_data = data_processor.preprocess_data(data)
                        X, y = data_processor.prepare_model_data(processed_data)
                        
                        # Split data
                        X_train, X_test, y_train, y_test = data_processor.split_data(
                            X, y, test_size=test_size, random_state=random_state
                        )
                        
                        # Train models
                        results = model_trainer.train_models(X_train, y_train, X_test, y_test)
                        
                        # Create visualizations
                        model_trainer.create_visualizations(
                            feature_columns=data_processor.feature_columns,
                            y_test=y_test
                        )
                        
                        # Save results in session state for persistence
                        st.session_state.training_results = results
                        st.session_state.best_model_name = model_trainer.best_model_name
                        st.session_state.model_metrics = results
                        st.session_state.training_completed = True
                        
                        # Save best model with PKL
                        metadata = {
                            'accuracy': results[model_trainer.best_model_name]['accuracy'],
                            'f1_score': results[model_trainer.best_model_name]['f1_score'],
                            'training_samples': len(X_train),
                            'test_samples': len(X_test),
                            'features_count': X.shape[1],
                            'best_model_name': model_trainer.best_model_name,
                            'uploaded_file': uploaded_file.name
                        }
                        
                        pkl_path = pkl_manager.save_model(
                            model=model_trainer.best_model,
                            model_name='healthcare_antibiotic',
                            model_type=model_trainer.best_model_name,
                            data_processor=data_processor,
                            metadata=metadata
                        )
                        
                        st.success("✅ Model training completed successfully!")
                        
                        # Show results
                        st.subheader("📊 Training Results")
                        
                        # Create results dataframe (excluding model objects)
                        results_display = {}
                        for model_name, metrics in results.items():
                            results_display[model_name] = {
                                'Accuracy': f"{metrics['accuracy']:.4f}",
                                'Precision': f"{metrics['precision']:.4f}",
                                'Recall': f"{metrics['recall']:.4f}",
                                'F1-Score': f"{metrics['f1_score']:.4f}"
                            }
                        
                        results_df = pd.DataFrame(results_display).T
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Show best model
                        st.info(f"🏆 Best Model: {model_trainer.best_model_name}")
                        
                        # Show metrics
                        col1, col2, col3, col4 = st.columns(4)
                        best_results = results[model_trainer.best_model_name]
                        
                        with col1:
                            st.metric("Accuracy", f"{best_results['accuracy']:.4f}")
                        with col2:
                            st.metric("Precision", f"{best_results['precision']:.4f}")
                        with col3:
                            st.metric("Recall", f"{best_results['recall']:.4f}")
                        with col4:
                            st.metric("F1-Score", f"{best_results['f1_score']:.4f}")
                        
                        # Save results (clean version without model objects)
                        results_for_json = {}
                        for model_name, metrics in results.items():
                            results_for_json[model_name] = {
                                'accuracy': metrics['accuracy'],
                                'precision': metrics['precision'],
                                'recall': metrics['recall'],
                                'f1_score': metrics['f1_score']
                            }
                        
                        results_file = f"results/training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        with open(results_file, 'w') as f:
                            json.dump(results_for_json, f, indent=2)
                        
                        st.success(f"📁 Results saved to: {results_file}")
                        st.success(f"💾 Model saved to: {pkl_path}")
                        
                    except Exception as e:
                        st.error(f"❌ Error during training: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    # Show uploaded file info if available in session state
    elif st.session_state.uploaded_file_name and st.session_state.file_data is not None:
        st.info(f"📁 Previously uploaded file: {st.session_state.uploaded_file_name}")
        st.write(f"**File Info:** {st.session_state.file_data.shape[0]} rows, {st.session_state.file_data.shape[1]} columns")
        
        # Show sample data
        st.subheader("📋 Sample Data")
        st.dataframe(st.session_state.file_data.head(), use_container_width=True)
        
        # Show training options
        st.subheader("⚙️ Training Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05, key="test_size_persist")
            random_state = st.number_input("Random State", 1, 100, 42, key="random_state_persist")
        
        with col2:
            models_to_train = st.multiselect(
                "Models to Train",
                ["XGBoost", "CatBoost", "Random Forest", "Logistic Regression"],
                default=["XGBoost", "CatBoost"],
                key="models_persist"
            )
        
        # Train button
        if st.button("🚀 Start Training", type="primary", use_container_width=True, key="train_persist"):
            file_path = os.path.join('data', st.session_state.uploaded_file_name)
            with st.spinner("Training models... This may take several minutes."):
                try:
                    # Load and preprocess data
                    data = data_processor.load_and_explore_data(file_path)
                    processed_data = data_processor.preprocess_data(data)
                    X, y = data_processor.prepare_model_data(processed_data)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = data_processor.split_data(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    
                    # Train models
                    results = model_trainer.train_models(X_train, y_train, X_test, y_test)
                    
                    # Create visualizations
                    model_trainer.create_visualizations(
                        feature_columns=data_processor.feature_columns,
                        y_test=y_test
                    )
                    
                    # Save results in session state for persistence
                    st.session_state.training_results = results
                    st.session_state.best_model_name = model_trainer.best_model_name
                    st.session_state.model_metrics = results
                    st.session_state.training_completed = True
                    
                    # Save best model with PKL
                    metadata = {
                        'accuracy': results[model_trainer.best_model_name]['accuracy'],
                        'f1_score': results[model_trainer.best_model_name]['f1_score'],
                        'training_samples': len(X_train),
                        'test_samples': len(X_test),
                        'features_count': X.shape[1],
                        'best_model_name': model_trainer.best_model_name,
                        'uploaded_file': st.session_state.uploaded_file_name
                    }
                    
                    pkl_path = pkl_manager.save_model(
                        model=model_trainer.best_model,
                        model_name='healthcare_antibiotic',
                        model_type=model_trainer.best_model_name,
                        data_processor=data_processor,
                        metadata=metadata
                    )
                    
                    st.success("✅ Model training completed successfully!")
                    
                    # Show results
                    st.subheader("📊 Training Results")
                    
                    # Create results dataframe (excluding model objects)
                    results_display = {}
                    for model_name, metrics in results.items():
                        results_display[model_name] = {
                            'Accuracy': f"{metrics['accuracy']:.4f}",
                            'Precision': f"{metrics['precision']:.4f}",
                            'Recall': f"{metrics['recall']:.4f}",
                            'F1-Score': f"{metrics['f1_score']:.4f}"
                        }
                    
                    results_df = pd.DataFrame(results_display).T
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Show best model
                    st.info(f"🏆 Best Model: {model_trainer.best_model_name}")
                    
                    # Show metrics
                    col1, col2, col3, col4 = st.columns(4)
                    best_results = results[model_trainer.best_model_name]
                    
                    with col1:
                        st.metric("Accuracy", f"{best_results['accuracy']:.4f}")
                    with col2:
                        st.metric("Precision", f"{best_results['precision']:.4f}")
                    with col3:
                        st.metric("Recall", f"{best_results['recall']:.4f}")
                    with col4:
                        st.metric("F1-Score", f"{best_results['f1_score']:.4f}")
                    
                    # Save results (clean version without model objects)
                    results_for_json = {}
                    for model_name, metrics in results.items():
                        results_for_json[model_name] = {
                            'accuracy': metrics['accuracy'],
                            'precision': metrics['precision'],
                            'recall': metrics['recall'],
                            'f1_score': metrics['f1_score']
                        }
                    
                    results_file = f"results/training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(results_file, 'w') as f:
                        json.dump(results_for_json, f, indent=2)
                    
                    st.success(f"📁 Results saved to: {results_file}")
                    st.success(f"💾 Model saved to: {pkl_path}")
                    
                except Exception as e:
                    st.error(f"❌ Error during training: {str(e)}")
