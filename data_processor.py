"""
Healthcare Data Processor
========================

Handles all data preprocessing, feature engineering, and data preparation
for the healthcare antibiotic classification model.
"""

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class HealthcareDataProcessor:
    """Handles all data preprocessing and feature engineering"""
    
    def __init__(self):
        self.label_encoders = {}
        self.feature_columns = []
    
    def load_and_explore_data(self, data_path):
        """Load data and perform basic exploration"""
        print("HEALTHCARE ANTIBIOTIC CLASSIFICATION PIPELINE")
        print("="*60)
        
        print(f"Loading data from: {data_path}")
        try:
            # Try different engines for Excel files
            engines_to_try = ['openpyxl', 'xlrd', None]
            
            for engine in engines_to_try:
                try:
                    if engine:
                        data = pd.read_excel(data_path, engine=engine)
                    else:
                        data = pd.read_excel(data_path)
                    print(f"Successfully loaded with engine: {engine}")
                    break
                except Exception as e:
                    print(f"Failed with engine {engine}: {str(e)}")
                    continue
            else:
                raise Exception("Could not read Excel file with any available engine")
            
            print(f"Data loaded successfully! Shape: {data.shape}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
        
        self._print_basic_info(data)
        return data
        
    
    def _print_basic_info(self, df):
        """Print basic data information"""
        print(f"\nDataset Shape: {df.shape}")
        
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            print(f"Missing values found in {missing_data[missing_data > 0].count()} columns")
    
    def preprocess_data(self, data):
        """Complete data preprocessing pipeline"""
        print("\nDATA PREPROCESSING")
        print("="*50)
        
        df = data.copy()
        
        df = self._handle_missing_values(df)
        df = self._create_features(df)
        df = self._encode_categorical_variables(df)
        
        if 'Result' not in df.columns:
            df['Result'] = self._create_target_variable(df)
        
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        print("Handling missing values...")
        
        # Handle datetime columns
        datetime_cols = ['EXAM_ORDER_DATE', 'EXAM_REPORT_DATE', 'ANTIBIOTIC_ORDER_DATE']
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[col] = df[col].ffill().bfill()
        
        # Handle categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Handle numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def _create_features(self, df):
        """Create engineered features"""
        print("Performing feature engineering...")
        
        # Extract date features
        date_cols = ['EXAM_ORDER_DATE', 'EXAM_REPORT_DATE', 'ANTIBIOTIC_ORDER_DATE']
        for col in date_cols:
            if col in df.columns:
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_dayofweek'] = df[col].dt.dayofweek
        
        # Extract features from EXAM_RESULT
        df = self._extract_exam_result_features(df)
        
        # Create antibiotic group features
        df = self._create_antibiotic_features(df)
        
        return df
    
    def _extract_exam_result_features(self, df):
        """Extract features from EXAM_RESULT text"""
        if 'EXAM_RESULT' not in df.columns:
            return df
        
        exam_result = df['EXAM_RESULT'].astype(str)
        
        # Basic text features
        df['organism_detected'] = exam_result.str.contains('bacteria|organism|isolated|detected', case=False, na=False).astype(int)
        df['susceptible_mentioned'] = exam_result.str.contains('susceptible|sensitive', case=False, na=False).astype(int)
        df['resistant_mentioned'] = exam_result.str.contains('resistant|resistance', case=False, na=False).astype(int)
        
        # Enhanced organism detection - comprehensive list based on your data
        organisms = [
            'pseudomonas', 'escherichia', 'klebsiella', 'staphylococcus', 'streptococcus', 
            'enterococcus', 'candida', 'aspergillus', 'acinetobacter', 'proteus', 
            'enterobacter', 'salmonella', 'shigella', 'haemophilus', 'neisseria', 
            'mycobacterium', 'clostridium', 'bacteroides', 'listeria', 'vibrio',
            'stenotrophomonas', 'saprochaete', 'yeast', 'fungi'
        ]
        
        # Create organism detection features
        for organism in organisms:
            df[f'organism_{organism}'] = exam_result.str.contains(organism, case=False, na=False).astype(int)
        
        # Additional specific organism patterns with abbreviations
        specific_organisms = {
            'e_coli': r'\b(?:escherichia\s+coli|e\.?\s*coli)\b',
            'k_pneumoniae': r'\b(?:klebsiella\s+pneumoniae|k\.?\s*pneumoniae)\b',
            'p_aeruginosa': r'\b(?:pseudomonas\s+aeruginosa|p\.?\s*aeruginosa)\b',
            's_aureus': r'\b(?:staphylococcus\s+aureus|s\.?\s*aureus)\b',
            's_epidermidis': r'\b(?:staphylococcus\s+epidermidis|s\.?\s*epidermidis)\b',
            's_pneumoniae': r'\b(?:streptococcus\s+pneumoniae|s\.?\s*pneumoniae)\b',
            'e_faecalis': r'\b(?:enterococcus\s+faecalis|e\.?\s*faecalis)\b',
            'c_albicans': r'\b(?:candida\s+albicans|c\.?\s*albicans)\b',
            'a_baumannii': r'\b(?:acinetobacter\s+baumannii|a\.?\s*baumannii)\b',
            'p_mirabilis': r'\b(?:proteus\s+mirabilis|p\.?\s*mirabilis)\b',
            'e_cloacae': r'\b(?:enterobacter\s+cloacae|e\.?\s*cloacae)\b',
            's_maltophilia': r'\b(?:stenotrophomonas\s+maltophilia|s\.?\s*maltophilia)\b',
            's_capitata': r'\b(?:saprochaete\s+capitata|s\.?\s*capitata)\b'
        }
        
        for org_name, pattern in specific_organisms.items():
            df[f'organism_{org_name}'] = exam_result.str.contains(pattern, case=False, na=False, regex=True).astype(int)
        
        # Text complexity features
        df['has_susceptibility_results'] = exam_result.str.contains('susceptible|resistant|sensitive', case=False, na=False).astype(int)
        df['exam_result_length'] = exam_result.str.len()
        df['exam_result_word_count'] = exam_result.str.split().str.len()
        
        return df
    
    def _create_antibiotic_features(self, df):
        """Create antibiotic-related features"""
        if 'ANTIBIOTIC_ORDER_NAME' not in df.columns:
            return df
        
        antibiotic_order = df['ANTIBIOTIC_ORDER_NAME'].astype(str)
        
        # Enhanced antibiotic classes - comprehensive list
        antibiotic_classes = {
            'penicillin': ['penicillin', 'amoxicillin', 'ampicillin', 'piperacillin', 'benzylpenicillin'],
            'cephalosporin': ['cef', 'cephalosporin', 'ceftazidime', 'ceftriaxone', 'cefepime', 'cefotaxime', 'cefuroxime', 'cefazolin'],
            'carbapenem': ['meropenem', 'imipenem', 'ertapenem', 'doripenem'],
            'aminoglycoside': ['gentamicin', 'tobramycin', 'amikacin', 'streptomycin'],
            'fluoroquinolone': ['ciprofloxacin', 'levofloxacin', 'moxifloxacin', 'norfloxacin', 'ofloxacin'],
            'glycopeptide': ['vancomycin', 'teicoplanin'],
            'antifungal': ['caspofungin', 'fluconazole', 'amphotericin', 'itraconazole', 'voriconazole'],
            'macrolide': ['azithromycin', 'clarithromycin', 'erythromycin'],
            'tetracycline': ['tetracycline', 'doxycycline', 'minocycline'],
            'sulfonamide': ['trimethoprim', 'sulfamethoxazole', 'sulfa'],
            'nitrofurantoin': ['nitrofurantoin'],
            'metronidazole': ['metronidazole', 'flagyl'],
            'colistin': ['colistin', 'polymyxin'],
            'linezolid': ['linezolid'],
            'daptomycin': ['daptomycin'],
            'tigecycline': ['tigecycline']
        }
        
        for class_name, antibiotics in antibiotic_classes.items():
            df[f'antibiotic_{class_name}'] = antibiotic_order.str.contains('|'.join(antibiotics), case=False, na=False).astype(int)
        
        # Extract dosage
        df['antibiotic_dosage'] = antibiotic_order.str.extract(r'(\d+(?:\.\d+)?)\s*(?:g|mg|mcg)', flags=re.IGNORECASE)
        df['antibiotic_dosage'] = pd.to_numeric(df['antibiotic_dosage'], errors='coerce')
        
        # Administration route
        df['antibiotic_injection'] = antibiotic_order.str.contains('injection|iv|intravenous', case=False, na=False).astype(int)
        
        return df
    
    def _encode_categorical_variables(self, df):
        """Encode categorical variables"""
        print("Encoding categorical variables...")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'Result':
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        return df
    
    def _create_target_variable(self, df):
        """Create target variable based on clinical rules"""
        print("Creating target variable based on clinical rules...")
        
        target = []
        
        for idx, row in df.iterrows():
            if pd.isna(row.get('EXAM_ABNORMAL_RESULT', None)) or str(row.get('EXAM_ABNORMAL_RESULT', '')).strip() == '':
                target.append('Mismatch')
                continue
            
            if str(row.get('EXAM_ABNORMAL_RESULT', '')).upper() == 'A':
                antibiotic_order = str(row.get('ANTIBIOTIC_ORDER_NAME', '')).lower()
                exam_result = str(row.get('EXAM_RESULT', '')).lower()
                
                if self._check_antibiotic_match(antibiotic_order, exam_result):
                    target.append('Match')
                else:
                    target.append('Mismatch')
            else:
                target.append('Mismatch')
        
        return target
    
    def _check_antibiotic_match(self, antibiotic_order, exam_result):
        """Check if antibiotic order matches susceptibility results"""
        antibiotic_name = self._extract_antibiotic_name(antibiotic_order)
        
        if antibiotic_name in exam_result:
            return True
        
        antibiotic_groups = {
            'penicillin': ['penicillin', 'amoxicillin', 'ampicillin', 'piperacillin', 'benzylpenicillin'],
            'cephalosporin': ['cef', 'cephalosporin', 'ceftazidime', 'ceftriaxone', 'cefepime', 'cefotaxime', 'cefuroxime', 'cefazolin'],
            'carbapenem': ['meropenem', 'imipenem', 'ertapenem', 'doripenem'],
            'aminoglycoside': ['gentamicin', 'tobramycin', 'amikacin', 'streptomycin'],
            'fluoroquinolone': ['ciprofloxacin', 'levofloxacin', 'moxifloxacin', 'norfloxacin', 'ofloxacin'],
            'glycopeptide': ['vancomycin', 'teicoplanin'],
            'antifungal': ['caspofungin', 'fluconazole', 'amphotericin', 'itraconazole', 'voriconazole'],
            'macrolide': ['azithromycin', 'clarithromycin', 'erythromycin'],
            'tetracycline': ['tetracycline', 'doxycycline', 'minocycline'],
            'sulfonamide': ['trimethoprim', 'sulfamethoxazole', 'sulfa'],
            'nitrofurantoin': ['nitrofurantoin'],
            'metronidazole': ['metronidazole', 'flagyl'],
            'colistin': ['colistin', 'polymyxin'],
            'linezolid': ['linezolid'],
            'daptomycin': ['daptomycin'],
            'tigecycline': ['tigecycline']
        }
        
        ordered_group = None
        for group, antibiotics in antibiotic_groups.items():
            if any(ab in antibiotic_name for ab in antibiotics):
                ordered_group = group
                break
        
        if ordered_group:
            for antibiotic in antibiotic_groups[ordered_group]:
                if antibiotic in exam_result:
                    return True
        
        return False
    
    def _extract_antibiotic_name(self, antibiotic_order):
        """Extract the main antibiotic name from the full order string"""
        antibiotic_name = re.sub(r'\d+(?:\.\d+)?\s*(?:g|mg|mcg)', '', antibiotic_order)
        antibiotic_name = re.sub(r'\s*-\s*injection.*$', '', antibiotic_name, flags=re.IGNORECASE)
        antibiotic_name = re.sub(r'\s*injection.*$', '', antibiotic_name, flags=re.IGNORECASE)
        antibiotic_name = re.sub(r'\s*tablet.*$', '', antibiotic_name, flags=re.IGNORECASE)
        antibiotic_name = re.sub(r'\s*capsule.*$', '', antibiotic_name, flags=re.IGNORECASE)
        antibiotic_name = re.sub(r'\s*\(.*?\)', '', antibiotic_name)
        antibiotic_name = re.sub(r'\s+', ' ', antibiotic_name).strip()
        
        return antibiotic_name
    
    def prepare_model_data(self, df):
        """Prepare final dataset for model training"""
        print("Preparing features and target...")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        feature_cols = [col for col in df.columns 
                       if col not in categorical_cols 
                       and col not in datetime_cols 
                       and col != 'Result']
        
        if 'RESULT_encoded' in feature_cols:
            feature_cols.remove('RESULT_encoded')
            print("Removed RESULT_encoded to prevent data leakage")
        
        X = df[feature_cols].fillna(0)
        y = (df['Result'] == 'Match').astype(int)
        
        self.feature_columns = feature_cols
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        print(f"Feature columns: {feature_cols}")
        
        if X.isnull().any().any():
            print("Warning: NaN values found. Filling with 0...")
            X = X.fillna(0)
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        print(f"\nSplitting data (test_size={test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
