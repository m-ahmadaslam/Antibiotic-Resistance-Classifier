
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os
from data_processor import HealthcareDataProcessor
from model_trainer import HealthcareModelTrainer

class HealthcareModelPKLManager:
    """Manages saving and loading healthcare models using PKL files"""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.data_processor = HealthcareDataProcessor()
        
        # Create models directory if it doesn't exist
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
    
    def save_model(self, model, model_name, model_type, data_processor=None, metadata=None):
        """
        Save model and related components to PKL file
        
        Args:
            model: Trained model object
            model_name: Name for the model
            model_type: Type of model (e.g., 'XGBoost', 'CatBoost')
            data_processor: Data processor with encoders
            metadata: Additional model information
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{model_type}_{timestamp}.pkl"
        filepath = os.path.join(self.models_dir, filename)
        
        # Prepare model package
        model_package = {
            'model': model,
            'model_type': model_type,
            'model_name': model_name,
            'data_processor': data_processor,
            'feature_columns': data_processor.feature_columns if data_processor else None,
            'label_encoders': data_processor.label_encoders if data_processor else {},
            'metadata': metadata or {},
            'timestamp': timestamp,
            'version': '1.0'
        }
        
        # Save to PKL file
        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"Model saved successfully: {filepath}")
        return filepath
    
    def load_model(self, model_path):
        """
        Load model from PKL file
        
        Args:
            model_path: Path to the PKL file
            
        Returns:
            Dictionary containing model and related components
        """
        try:
            with open(model_path, 'rb') as f:
                model_package = pickle.load(f)
            
            print(f"Model loaded successfully: {model_path}")
            print(f"Model type: {model_package['model_type']}")
            print(f"Saved on: {model_package['timestamp']}")
            
            return model_package
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
    
    def predict_with_pkl_model(self, model_path, new_data_path, output_path=None):
        """
        Make predictions using a saved PKL model
        
        Args:
            model_path: Path to the PKL model file
            new_data_path: Path to new data file (Excel)
            output_path: Path to save predictions (optional)
        """
        print("PREDICTING WITH PKL MODEL")
        print("="*40)
        
        # Load the model
        model_package = self.load_model(model_path)
        if not model_package:
            return None
        
        # Load new data
        print(f"Loading new data from: {new_data_path}")
        
        # Try different engines for Excel files
        engines_to_try = ['openpyxl', 'xlrd', None]
        
        for engine in engines_to_try:
            try:
                if engine:
                    new_data = pd.read_excel(new_data_path, engine=engine)
                else:
                    new_data = pd.read_excel(new_data_path)
                print(f"Successfully loaded with engine: {engine}")
                break
            except Exception as e:
                print(f"Failed with engine {engine}: {str(e)}")
                continue
        else:
            raise Exception("Could not read Excel file with any available engine")
        
        print(f"New data shape: {new_data.shape}")
        
        # Preprocess new data using the same processor
        data_processor = model_package['data_processor']
        if data_processor:
            # Apply the same preprocessing
            processed_data = data_processor.preprocess_data(new_data)
            X_new, y_new = data_processor.prepare_model_data(processed_data)
        else:
            print("Warning: No data processor found in model package")
            return None
        
        # Make predictions
        model = model_package['model']
        predictions = model.predict(X_new)
        probabilities = model.predict_proba(X_new) if hasattr(model, 'predict_proba') else None
        
        # Create results dataframe
        results = new_data.copy()
        results['Predicted_Result'] = predictions
        results['Prediction_Probability'] = probabilities[:, 1] if probabilities is not None else None
        results['Prediction_Class'] = ['Match' if p == 1 else 'Mismatch' for p in predictions]
        
        # Save results if output path provided
        if output_path:
            # Convert .xls to .xlsx for compatibility
            if output_path.endswith('.xls'):
                output_path = output_path.replace('.xls', '.xlsx')
            
            # Use openpyxl engine for writing Excel files
            results.to_excel(output_path, index=False, engine='openpyxl')
            print(f"Predictions saved to: {output_path}")
        
        # Print summary
        print(f"\nPrediction Summary:")
        print(f"Total samples: {len(predictions)}")
        print(f"Predicted Matches: {sum(predictions)}")
        print(f"Predicted Mismatches: {len(predictions) - sum(predictions)}")
        
        return results
    
    def list_saved_models(self):
        """List all saved models in the models directory"""
        print("SAVED MODELS")
        print("="*30)
        
        if not os.path.exists(self.models_dir):
            print("No models directory found")
            return []
        
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]
        
        if not model_files:
            print("No saved models found")
            return []
        
        for i, filename in enumerate(model_files, 1):
            filepath = os.path.join(self.models_dir, filename)
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            
            print(f"{i}. {filename}")
            print(f"   Size: {file_size:.2f} MB")
            
            # Try to load basic info
            try:
                with open(filepath, 'rb') as f:
                    model_package = pickle.load(f)
                    print(f"   Type: {model_package.get('model_type', 'Unknown')}")
                    print(f"   Saved: {model_package.get('timestamp', 'Unknown')}")
            except:
                print(f"   Error reading model info")
            print()
        
        return model_files
    
    def compare_models(self, model_paths):
        """
        Compare multiple saved models
        
        Args:
            model_paths: List of paths to PKL model files
        """
        print("MODEL COMPARISON")
        print("="*30)
        
        models_info = []
        
        for path in model_paths:
            model_package = self.load_model(path)
            if model_package:
                models_info.append({
                    'path': path,
                    'type': model_package['model_type'],
                    'timestamp': model_package['timestamp'],
                    'model': model_package['model']
                })
        
        # Print comparison table
        print(f"{'Model Type':<15} {'Saved Date':<15} {'File Size (MB)':<15}")
        print("-" * 50)
        
        for info in models_info:
            file_size = os.path.getsize(info['path']) / (1024 * 1024)
            print(f"{info['type']:<15} {info['timestamp']:<15} {file_size:<15.2f}")

def main():
    """Example usage of PKL manager"""
    
    # Initialize PKL manager
    pkl_manager = HealthcareModelPKLManager()
    
    # List existing models
    pkl_manager.list_saved_models()
    
    # Example: Train and save a new model
    print("\nTRAINING AND SAVING NEW MODEL")
    print("="*40)
    
    # Load and preprocess data
    data_processor = HealthcareDataProcessor()
    data = data_processor.load_and_explore_data('data/Drug_Bug_Mismatch_Aspupdated.xls')
    processed_data = data_processor.preprocess_data(data)
    X, y = data_processor.prepare_model_data(processed_data)
    
    # Train model
    trainer = HealthcareModelTrainer()
    X_train, X_test, y_train, y_test = data_processor.split_data(X, y)
    
    # Train XGBoost model
    xgb_model = trainer.train_xgboost_model(X_train, y_train)
    
    # Save model with PKL
    metadata = {
        'accuracy': trainer.evaluate_model(xgb_model, X_test, y_test),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'features_count': X.shape[1]
    }
    
    model_path = pkl_manager.save_model(
        model=xgb_model,
        model_name='healthcare_antibiotic',
        model_type='XGBoost',
        data_processor=data_processor,
        metadata=metadata
    )
    
    print(f"Model saved to: {model_path}")
    
    # Example: Load and use saved model
    print("\nLOADING AND USING SAVED MODEL")
    print("="*40)
    
    loaded_package = pkl_manager.load_model(model_path)
    if loaded_package:
        print("Model loaded successfully!")
        print(f"Model type: {loaded_package['model_type']}")
        print(f"Features: {len(loaded_package['feature_columns'])}")

if __name__ == "__main__":
    main()
