# 🏥 Healthcare Antibiotic Classification System

A complete machine learning system for classifying antibiotic-organism matches in healthcare data, featuring a user-friendly Streamlit interface and AI chatbot.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run app.py
```

### 3. Open in Browser
Navigate to `http://localhost:8501`

## ✨ Features

### 📊 **Complete ML Pipeline**
- **Enhanced Data Processing**: 25+ organisms, 16 antibiotic classes
- **Multiple ML Models**: XGBoost, CatBoost, Random Forest, Logistic Regression
- **Automatic Feature Engineering**: Organism detection, antibiotic classification
- **PKL Model Storage**: Fast loading and deployment

### 🖥️ **User-Friendly Interface**
- **Dashboard**: System overview and quick actions
- **File Upload**: Drag-and-drop Excel file upload
- **Model Training**: One-click model training with options
- **Predictions**: Instant predictions on new data
- **Results Visualization**: Charts and downloadable results

### 🤖 **AI Healthcare Chatbot**
- **Antibiotic Information**: Detailed drug information and uses
- **Organism Details**: Common pathogens and treatments
- **Match/Mismatch Guidance**: Understanding drug-bug relationships
- **Quick Actions**: Pre-built queries for common questions

### 📁 **Model Management**
- **PKL File Management**: View, download, and delete models
- **Performance Tracking**: Accuracy, F1-score, and metadata
- **Version Control**: Timestamped model versions
- **Feature Analysis**: Detailed feature importance

## 📋 System Requirements

- **Python**: 3.8+
- **Memory**: 4GB+ RAM recommended
- **Storage**: 1GB+ free space
- **Browser**: Modern web browser (Chrome, Firefox, Safari, Edge)

## 📊 Data Format

Your Excel file must contain these columns:

| Column | Required | Description |
|--------|----------|-------------|
| `EXAM_ORDER_DATE` | ✅ | Date of exam order |
| `EXAM_REPORT_DATE` | ✅ | Date of exam report |
| `EXAM_RESULT` | ✅ | Text result with organism details |
| `EXAM_ABNORMAL_RESULT` | ✅ | A for abnormal, N for normal |
| `ANTIBIOTIC_ORDER_DATE` | ✅ | Date of antibiotic order |
| `ANTIBIOTIC_ORDER_NAME` | ✅ | Name of prescribed antibiotic |
| `EXAM_ORDER_WARD_CD` | ✅ | Ward code |
| `ANTIBIOTIC_ORDER_WARD` | ✅ | Ward where antibiotic ordered |
| `ORDERING_DEPARTMENT_CD` | ✅ | Department code |

## 🎯 Usage Guide

### **Step 1: Upload & Train**
1. Go to "📁 Upload & Train"
2. Upload your Excel file
3. Configure training options
4. Click "Start Training"
5. Wait for completion (5-15 minutes)

### **Step 2: Make Predictions**
1. Go to "🔮 Make Predictions"
2. Select a trained model
3. Upload new data file
4. Get instant predictions
5. Download results

### **Step 3: Use Chatbot**
1. Go to "🤖 AI Chatbot"
2. Ask questions about:
   - Specific antibiotics (ceftriaxone, meropenem, vancomycin)
   - Organisms (E. coli, Klebsiella, Pseudomonas)
   - Drug-bug matching concepts
   - Treatment guidelines

## 🔧 Technical Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   Data Processor│    │   Model Trainer │
│                 │    │                 │    │                 │
│ • File Upload   │───▶│ • Preprocessing │───▶│ • XGBoost       │
│ • Chatbot       │    │ • Feature Eng.  │    │ • CatBoost      │
│ • Visualizations│    │ • Organism Det. │    │ • Random Forest │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  PKL Manager    │    │   Results &     │
                       │                 │    │   Visualizations│
                       │ • Save Models   │    │                 │
                       │ • Load Models   │    │ • Confusion Mat.│
                       │ • Predictions   │    │ • Feature Imp.  │
                       └─────────────────┘    │ • Performance   │
                                              └─────────────────┘
```

## 🚀 Deployment Options

### **Local Development**
```bash
streamlit run app.py
```

### **Streamlit Cloud**
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Deploy automatically

### **Docker**
```bash
docker build -t healthcare-app .
docker run -p 8501:8501 healthcare-app
```

## 📈 Performance

### **Enhanced Features**
- **25+ Organisms**: Comprehensive organism detection
- **16 Antibiotic Classes**: Complete antibiotic classification
- **Automatic Preprocessing**: No manual feature engineering needed
- **PKL Storage**: 2-3 second model loading vs 10-30 minutes training

### **Model Performance**
- **Accuracy**: 85-95% typical performance
- **F1-Score**: 0.80-0.90 range
- **Training Time**: 5-15 minutes depending on data size
- **Prediction Time**: <1 second per sample

## 🛠️ Development

### **Project Structure**
```
healthcare/
├── app.py                 # Main Streamlit application
├── data_processor.py      # Enhanced data preprocessing
├── model_trainer.py       # ML model training
├── model_pkl_manager.py   # PKL model management
├── quick_predict.py       # Command-line predictions
├── main.py               # CLI interface
├── requirements.txt      # Dependencies
├── README.md            # This file
├── DEPLOYMENT_GUIDE.md  # Deployment instructions
├── data/                # Uploaded data files
├── models/              # Saved PKL models
├── results/             # Training results
└── predictions/         # Prediction outputs
```

### **Key Components**

#### **Data Processor (`data_processor.py`)**
- Enhanced organism detection (25+ organisms)
- Comprehensive antibiotic classification (16 classes)
- Automatic feature engineering
- Missing value handling

#### **Model Trainer (`model_trainer.py`)**
- Multiple ML algorithms
- Hyperparameter optimization
- Performance evaluation
- Visualization generation

#### **PKL Manager (`model_pkl_manager.py`)**
- Model serialization/deserialization
- Fast prediction pipeline
- Model versioning
- Metadata storage

#### **Streamlit App (`app.py`)**
- Complete web interface
- File upload handling
- Real-time training progress
- Interactive visualizations
- AI chatbot integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues or questions:
1. Check the troubleshooting section in `DEPLOYMENT_GUIDE.md`
2. Review data format requirements
3. Ensure all dependencies are installed
4. Check system requirements

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎉 Acknowledgments

- **Healthcare Data**: Real-world antibiotic classification data
- **Streamlit**: For the amazing web framework
- **Scikit-learn**: For machine learning algorithms
- **XGBoost & CatBoost**: For high-performance models

---

**🏥 Built for Healthcare Professionals**  
**🤖 Powered by Machine Learning**  
**⚡ Optimized for Speed and Accuracy**
