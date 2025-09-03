"""
Healthcare Model Trainer
========================
Handles all machine learning model training, evaluation, and prediction
for the healthcare antibiotic classification model.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from catboost import CatBoostClassifier
import joblib
import os
from datetime import datetime

class HealthcareModelTrainer:
    """Handles all ML model training and evaluation"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        self.feature_importance = None
    
    def handle_class_imbalance(self, X_train, y_train):
        """Handle class imbalance using SMOTE"""
        print("Handling class imbalance...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        return X_train_balanced, y_train_balanced
    
    def initialize_models(self):
        """Initialize top 3 ML models for comparison"""
        self.models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'CatBoost': CatBoostClassifier(random_state=42, verbose=False)
        }
        print(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train all models and compare performance"""
        print("\nMODEL TRAINING AND EVALUATION")
        print("="*50)
        
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(X_train, y_train)
        self.initialize_models()
        
        best_score = 0
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            try:
                model.fit(X_train_balanced, y_train_balanced)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                self.results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                if f1 > best_score:
                    best_score = f1
                    self.best_model = model
                    self.best_model_name = name
                
                print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        print(f"Best Model: {self.best_model_name} (F1: {best_score:.4f})")
        return self.results
    
    def evaluate_best_model(self, X_test, y_test, label_encoders):
        """Detailed evaluation of the best model"""
        if not self.best_model:
            print("No best model found!")
            return
        
        print(f"\nDETAILED EVALUATION - {self.best_model_name}")
        print("="*50)
        
        y_pred = self.best_model.predict(X_test)
        target_names = ['Mismatch', 'Match']  # Binary classification
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        self._extract_feature_importance()
    
    def _extract_feature_importance(self):
        """Extract and display feature importance"""
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            self.feature_importance = np.abs(self.best_model.coef_[0])
        else:
            print("Feature importance not available for this model")
    
    def create_visualizations(self, feature_columns, y_test=None):
        """Create and save visualizations"""
        print("\nCreating visualizations...")
        
        try:
            self._plot_model_comparison()
            
            if self.feature_importance is not None:
                self._plot_feature_importance(feature_columns)
            
            if y_test is not None:
                y_pred = self.results[self.best_model_name]['predictions']
                target_names = ['Match', 'Mismatch']
                self._plot_confusion_matrix(y_test, y_pred, class_names=target_names)
            
            print("Visualizations saved to results/ directory")
            
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
    
    def _plot_model_comparison(self):
        """Plot model performance comparison"""
        if not self.results:
            return
        
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Model Performance Comparison', fontsize=14)
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            values = [self.results[model][metric] for model in models]
            bars = ax.bar(models, values, color='skyblue', alpha=0.7)
            ax.set_title(f'{metric.title()} Comparison')
            ax.set_ylabel(metric.title())
            ax.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, feature_columns):
        """Plot feature importance"""
        if self.feature_importance is None:
            return
        
        indices = np.argsort(self.feature_importance)[-20:]
        top_features = [feature_columns[i] for i in indices]
        top_importance = self.feature_importance[indices]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_importance, color='lightcoral', alpha=0.7)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top 20 Feature Importance - {self.best_model_name}')
        plt.grid(axis='x', alpha=0.3)
        
        for i, (feature, importance) in enumerate(zip(top_features, top_importance)):
            plt.text(importance + 0.001, i, f'{importance:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nTop 10 Most Important Features:")
        for i, (feature, importance) in enumerate(zip(top_features[-10:], top_importance[-10:]), 1):
            print(f"{i:2d}. {feature}: {importance:.4f}")
    
    def _plot_confusion_matrix(self, y_test, y_pred, class_names=None):
        """Plot confusion matrix for best model"""
        if not self.best_model or self.best_model_name not in self.results:
            return
        
        cm = confusion_matrix(y_test, y_pred)
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        accuracy = accuracy_score(y_test, y_pred)
        plt.text(0.02, 0.98, f'Accuracy: {accuracy:.3f}', 
                transform=plt.gca().transAxes, fontsize=12, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion Matrix Analysis for {self.best_model_name}:")
        for i, true_class in enumerate(class_names):
            for j, pred_class in enumerate(class_names):
                if cm[i, j] > 0:
                    print(f"True {true_class} → Predicted {pred_class}: {cm[i, j]} cases")
    
    def save_model(self, model_name=None):
        """Save the best trained model"""
        if not self.best_model:
            print("No model to save!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"healthcare_model_{self.best_model_name}_{timestamp}.pkl"
        filepath = os.path.join('models', filename)
        
        joblib.dump(self.best_model, filepath)
        
        performance_data = self.results[self.best_model_name].copy()
        performance_data.pop('model', None)
        performance_data.pop('predictions', None)
        performance_data.pop('probabilities', None)
        
        summary = {
            'model_name': self.best_model_name,
            'performance': performance_data,
            'timestamp': timestamp
        }
        
        summary_path = os.path.join('results', f'model_summary_{timestamp}.json')
        import json
        with open(summary_path, 'w') as f:
            json_summary = {k: (v.tolist() if isinstance(v, np.ndarray) else 
                               float(v) if isinstance(v, np.floating) else 
                               int(v) if isinstance(v, np.integer) else v) 
                           for k, v in summary.items()}
            json.dump(json_summary, f, indent=2)
        
        print(f"Model saved: {filepath}")
        print(f"Summary saved: {summary_path}")
        
        return filepath
    
    def predict_new_data(self, X_new):
        """Make predictions on new data"""
        if not self.best_model:
            print("No trained model available!")
            return None
        
        predictions = self.best_model.predict(X_new)
        probabilities = None
        
        if hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(X_new)
        
        return predictions, probabilities
