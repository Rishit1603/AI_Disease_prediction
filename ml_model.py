import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.base import clone
import joblib  # For saving/loading models

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class DiseasePredictor:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_dir = self.base_dir  # Look in the same directory as ml_model.py
        self.models_dir = os.path.join(self.base_dir, 'trained_models')
        
        # Create models directory if it doesn't exist
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
        self.load_datasets()
        self.vectorizer = TfidfVectorizer(max_features=50)
        
        # Create an ensemble of models
        self.svm = SVC(kernel='linear', probability=True, random_state=42)
        self.gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        self.ensemble = VotingClassifier(
            estimators=[
                ('svm', self.svm),
                ('gb', self.gb)
            ],
            voting='soft'
        )
        
        self.label_encoder = LabelEncoder()
        
        # Try to load trained models first
        if not self.load_trained_models():
            print("Training new models...")
            self._prepare_model()
            self.save_trained_models()

    def save_trained_models(self):
        """Save trained models and related components"""
        try:
            # Save models
            joblib.dump(self.svm, os.path.join(self.models_dir, 'svm_model.pkl'))
            joblib.dump(self.gb, os.path.join(self.models_dir, 'gb_model.pkl'))
            joblib.dump(self.ensemble, os.path.join(self.models_dir, 'ensemble_model.pkl'))
            joblib.dump(self.vectorizer, os.path.join(self.models_dir, 'vectorizer.pkl'))
            joblib.dump(self.label_encoder, os.path.join(self.models_dir, 'label_encoder.pkl'))
            
            # Save additional data
            with open(os.path.join(self.models_dir, 'model_data.json'), 'w') as f:
                json.dump({
                    'unique_symptoms': self.unique_symptoms,
                    'disease_list': self.disease_list.tolist() if hasattr(self, 'disease_list') else [],
                    'symptom_severity_map': self.symptom_severity_map
                }, f)
            print("Models saved successfully")
            return True
        except Exception as e:
            print(f"Error saving models: {e}")
            return False

    def load_trained_models(self):
        """Load trained models if they exist"""
        try:
            model_files = [
                'svm_model.pkl',
                'gb_model.pkl',
                'ensemble_model.pkl',
                'vectorizer.pkl',
                'label_encoder.pkl',
                'model_data.json'
            ]
            
            # Check if all model files exist
            if not all(os.path.exists(os.path.join(self.models_dir, f)) for f in model_files):
                return False
            
            # Load models
            self.svm = joblib.load(os.path.join(self.models_dir, 'svm_model.pkl'))
            self.gb = joblib.load(os.path.join(self.models_dir, 'gb_model.pkl'))
            self.ensemble = joblib.load(os.path.join(self.models_dir, 'ensemble_model.pkl'))
            self.vectorizer = joblib.load(os.path.join(self.models_dir, 'vectorizer.pkl'))
            self.label_encoder = joblib.load(os.path.join(self.models_dir, 'label_encoder.pkl'))
            
            # Load additional data
            with open(os.path.join(self.models_dir, 'model_data.json'), 'r') as f:
                data = json.load(f)
                self.unique_symptoms = data['unique_symptoms']
                self.disease_list = np.array(data['disease_list'])
                self.symptom_severity_map = data['symptom_severity_map']
            
            print("Models loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def load_datasets(self):
        """Load all required datasets"""
        try:
            # Define file paths - look in current directory instead of dataset subdirectory
            dataset_path = 'dataset.csv'
            severity_path = 'Symptom-severity.csv'
            desc_path = 'symptom_Description.csv'
            precaution_path = 'symptom_precaution.csv'

            # Check if files exist
            if not os.path.exists(dataset_path):
                print(f"Dataset file not found at {dataset_path}")
                self._use_fallback_data()
                return

            # Load main dataset with new format
            self.dataset = pd.read_csv(dataset_path)

            # Create symptom descriptions from Treatments column
            self.symptom_desc = pd.DataFrame({
                'Disease': self.dataset['Name'],
                'Description': self.dataset['Treatments']
            })

            # Create precautions from Treatments
            treatments = self.dataset['Treatments'].str.split(',', expand=True)
            max_treatments = len(treatments.columns)
            precaution_data = {
                'Disease': self.dataset['Name']
            }

            for i in range(min(max_treatments, 4)):  # Use up to 4 treatments
                precaution_data[f'Precaution_{i + 1}'] = treatments[i] if i < max_treatments else ''
            self.precautions = pd.DataFrame(precaution_data)

            # Extract unique symptoms and create severity data
            all_symptoms = []
            for symptoms in self.dataset['Symptoms']:
                if isinstance(symptoms, str):
                    all_symptoms.extend([s.strip().lower() for s in symptoms.split(',')])

            unique_symptoms = list(set(all_symptoms))

            # Create basic severity weights (can be adjusted based on frequency)
            self.severity_data = pd.DataFrame({
                'Symptom': unique_symptoms,
                'weight': [5] * len(unique_symptoms)  # Default medium severity
            })

            # Process datasets
            self._process_datasets()
            print("Successfully loaded all datasets")

        except Exception as e:
            print(f"Error loading datasets: {e}")
            self._use_fallback_data()
    def _use_fallback_data(self):
        """Use basic fallback data when datasets are not available"""
        print("Using fallback data...")
        self.disease_symptoms = {
            "Common Cold": ["fever", "cough", "sore throat", "runny nose", "congestion"],
            "Flu": ["fever", "cough", "sore throat", "body aches", "fatigue", "headache"],
            "Allergies": ["sneezing", "runny nose", "itchy eyes", "congestion"],
            "Migraine": ["headache", "nausea", "sensitivity to light", "sensitivity to sound"],
            "Sinusitis": ["facial pain", "congestion", "headache", "postnasal drip"]
        }
        
        # Create simple severity data
        self.severity_data = pd.DataFrame({
            'Symptom': ['fever', 'cough', 'headache', 'fatigue'],
            'weight': [7, 4, 5, 3]
        })
        
        # Create simple precautions
        self.precautions = pd.DataFrame({
            'Disease': list(self.disease_symptoms.keys()),
            'Precaution_1': ['Rest', 'Rest', 'Avoid allergens', 'Rest in dark room', 'Nasal irrigation'],
            'Precaution_2': ['Hydration', 'Medication', 'Take antihistamines', 'Stay hydrated', 'Stay hydrated'],
            'Precaution_3': ['Over-the-counter medicine', 'Isolation', 'Use air purifier', 'Avoid triggers', 'Use humidifier'],
            'Precaution_4': ['Monitor symptoms', 'Seek medical help', 'Consult doctor', 'Medication', 'Consult doctor']
        })
        
        # Create simple symptom descriptions
        self.symptom_desc = pd.DataFrame({
            'Disease': list(self.disease_symptoms.keys()),
            'Description': [
                'A viral infection causing upper respiratory symptoms',
                'A more severe viral infection with systemic symptoms',
                'An immune response to environmental triggers',
                'A neurological condition causing severe headaches',
                'Inflammation of the sinus cavities'
            ]
        })

    def _process_datasets(self):
        """Process and prepare the datasets with improved feature engineering for new format"""
        # Create a mapping of symptoms to their severity weights
        self.symptom_severity_map = dict(zip(
            self.severity_data['Symptom'],
            self.severity_data['weight']
        ))
        
        # Create a combined symptoms text for each disease with severity weights
        self.disease_symptoms = {}
        self.symptom_vectors = []
        self.diseases = []
        
        # Get unique symptoms from the dataset and clean them
        unique_symptoms = set()
        for symptoms in self.dataset['Symptoms']:
            if isinstance(symptoms, str):
                for symptom in symptoms.split(','):
                    cleaned_symptom = symptom.strip().lower()
                    if cleaned_symptom:
                        unique_symptoms.add(cleaned_symptom)
        
        # Convert to sorted list for consistent ordering
        self.unique_symptoms = sorted(list(unique_symptoms))
        
        # Process each disease
        for _, row in self.dataset.iterrows():
            disease = row['Name']
            symptoms_text = row['Symptoms']
            
            if not isinstance(symptoms_text, str):
                continue
                
            symptoms = [s.strip() for s in symptoms_text.split(',')]
            cleaned_symptoms = [s.lower() for s in symptoms if s.strip()]
            
            # Create a zero vector for all possible symptoms
            symptom_vector = np.zeros(len(self.unique_symptoms))
            
            # Set vector values for present symptoms
            for symptom in cleaned_symptoms:
                try:
                    idx = self.unique_symptoms.index(symptom)
                    weight = self.symptom_severity_map.get(symptom, 1)
                    symptom_vector[idx] = weight
                except ValueError:
                    continue
            
            if cleaned_symptoms:
                # Normalize the vector if possible
                if np.max(symptom_vector) > 0:
                    symptom_vector = symptom_vector / np.max(symptom_vector)
                
                # Add to datasets
                self.disease_symptoms[disease] = cleaned_symptoms
                self.symptom_vectors.append(symptom_vector)
                self.diseases.append(disease)

    def _prepare_model(self):
        """Prepare and train the ensemble model with improved features and multiple epochs"""
        try:
            if not self.disease_symptoms:
                raise ValueError("No disease symptoms data available")
            
            print("\n=== Starting Model Training ===")
            print("1. Preparing training data with augmentation...")
            
            # Prepare text data with augmentation
            X_text = []
            y = []
            
            # Count samples per disease for handling imbalanced classes
            disease_counts = {}
            for disease, symptoms in self.disease_symptoms.items():
                if disease not in disease_counts:
                    disease_counts[disease] = 0
                disease_counts[disease] += 1
            
            # Create symptom severity weights
            severity_weights = {}
            for symptom in self.unique_symptoms:
                severity_weights[symptom] = self.symptom_severity_map.get(symptom, 1)
            
            for disease, symptoms in self.disease_symptoms.items():
                # Original symptoms with severity weights
                weighted_symptoms = []
                for symptom in symptoms:
                    weight = severity_weights.get(symptom, 1)
                    weighted_symptoms.extend([symptom] * weight)
                
                X_text.append(" ".join(weighted_symptoms))
                y.append(disease)
                
                # Add variations for better training
                if len(symptoms) > 1:
                    # Add partial combinations with severity weights
                    for i in range(len(symptoms)):
                        partial_symptoms = symptoms[:i] + symptoms[i+1:]
                        weighted_partial = []
                        for symptom in partial_symptoms:
                            weight = severity_weights.get(symptom, 1)
                            weighted_partial.extend([symptom] * weight)
                        X_text.append(" ".join(weighted_partial))
                        y.append(disease)
            
            print(f"Total training samples: {len(X_text)}")
            print(f"Number of unique diseases: {len(set(y))}")
            
            print("\n2. Converting text to features...")
            # Convert text to TF-IDF features with improved parameters
            self.vectorizer = TfidfVectorizer(
                max_features=100,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.95,
                norm='l2'
            )
            
            X_tfidf = self.vectorizer.fit_transform(X_text)
            print(f"Feature matrix shape: {X_tfidf.shape}")
            
            # Encode labels
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Split data with stratification
            X_train, X_val, y_train, y_val = train_test_split(
                X_tfidf, y_encoded,
                test_size=0.4,  # Changed from 0.2 to 0.4 (60/40 split)
                random_state=42,
                stratify=y_encoded
            )
            
            print("\n3. Initializing models...")
            # Create improved models with adjusted parameters
            self.svm = SVC(
                kernel='rbf',
                probability=True,
                C=0.8,  # Reduced from 1.0 to prevent overfitting
                gamma='auto',  # Changed from 'scale' to 'auto' for better generalization
                class_weight='balanced',
                random_state=42
            )
            
            self.gb = GradientBoostingClassifier(
                n_estimators=120,  # Reduced from 150
                learning_rate=0.08,  # Reduced from 0.1
                max_depth=3,
                min_samples_split=5,  # Increased from 4
                min_samples_leaf=3,  # Increased from 2
                subsample=0.7,  # Reduced from 0.8
                max_features='sqrt',
                random_state=42
            )
            
            # Initialize best models
            best_svm = None
            best_gb = None
            best_val_score = 0
            patience = 3  # Increased patience for better model selection
            patience_counter = 0
            
            # Training with multiple epochs
            n_epochs = 8  # Reduced from 10 to prevent overfitting
            print(f"\n4. Starting training for {n_epochs} epochs...")
            
            for epoch in range(n_epochs):
                print(f"\nEpoch {epoch + 1}/{n_epochs}:")
                
                # Train SVM
                print("  Training SVM...")
                self.svm.fit(X_train, y_train)
                svm_train_score = self.svm.score(X_train, y_train)
                svm_val_score = self.svm.score(X_val, y_val)
                print(f"    SVM - Train: {svm_train_score:.4f}, Val: {svm_val_score:.4f}")
                
                # Train GradientBoosting
                print("  Training Gradient Boosting...")
                self.gb.fit(X_train, y_train)
                gb_train_score = self.gb.score(X_train, y_train)
                gb_val_score = self.gb.score(X_val, y_val)
                print(f"    GB - Train: {gb_train_score:.4f}, Val: {gb_val_score:.4f}")
                
                # Create and train ensemble
                print("  Training Ensemble...")
                self.ensemble = VotingClassifier(
                    estimators=[
                        ('svm', self.svm),
                        ('gb', self.gb)
                    ],
                    voting='soft',
                    weights=[0.4, 0.6]
                )
                
                self.ensemble.fit(X_train, y_train)
                ensemble_val_score = self.ensemble.score(X_val, y_val)
                print(f"    Ensemble - Val: {ensemble_val_score:.4f}")
                
                # Save best models
                if ensemble_val_score > best_val_score:
                    best_val_score = ensemble_val_score
                    best_svm = clone(self.svm)
                    best_gb = clone(self.gb)
                    patience_counter = 0
                    print(f"    New best validation score: {best_val_score:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("    Early stopping triggered")
                        break
            
            # Use best models if found
            if best_svm is not None and best_gb is not None:
                self.svm = best_svm
                self.gb = best_gb
                self.ensemble = VotingClassifier(
                    estimators=[
                        ('svm', self.svm),
                        ('gb', self.gb)
                    ],
                    voting='soft',
                    weights=[0.4, 0.6]
                )
            
            print("\n5. Final training on full dataset...")
            # Final training on full dataset
            self.svm.fit(X_tfidf, y_encoded)
            self.gb.fit(X_tfidf, y_encoded)
            self.ensemble.fit(X_tfidf, y_encoded)
            
            # Store the disease list
            self.disease_list = self.label_encoder.classes_
            print("\n=== Model training completed successfully ===")
            print(f"Final validation score: {best_val_score:.4f}")
            
        except Exception as e:
            print(f"\nError preparing model: {str(e)}")
            print("Debug info:")
            print(f"Number of diseases: {len(self.disease_symptoms) if hasattr(self, 'disease_symptoms') else 'N/A'}")
            print(f"Number of unique symptoms: {len(self.unique_symptoms) if hasattr(self, 'unique_symptoms') else 'N/A'}")
            self._initialize_basic_model()

    def _initialize_basic_model(self):
        """Initialize a basic model when the main model fails"""
        print("Initializing basic model...")
        
        # Create simple feature vector
        X = []
        y = []
        
        for disease, symptoms in self.disease_symptoms.items():
            # Original symptoms
            X.append(" ".join(symptoms))
            y.append(disease)
            
            # Add synthetic samples for better training
            if len(symptoms) > 1:
                # Add partial combinations
                for i in range(len(symptoms)):
                    partial_symptoms = symptoms[:i] + symptoms[i+1:]
                    X.append(" ".join(partial_symptoms))
                    y.append(disease)
        
        # Use the same vectorizer settings as the main model
        self.vectorizer = TfidfVectorizer(
            max_features=50,  # Match the feature size of the main model
            ngram_range=(1, 2),
            stop_words='english',
            min_df=1
        )
        X_vec = self.vectorizer.fit_transform(X)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train models with the same feature dimensions
        try:
            self.svm.fit(X_vec, y_encoded)
            self.gb.fit(X_vec, y_encoded)
            self.ensemble.fit(X_vec, y_encoded)
            self.disease_list = self.label_encoder.classes_
            print("Basic model initialized successfully")
        except Exception as e:
            print(f"Error training basic model: {e}")
            self.disease_list = list(self.disease_symptoms.keys())

    def _preprocess_text(self, text):
        """Preprocess the input text with improved tokenization"""
        # Convert to lowercase
        text = text.lower()
        
        # Split on commas and spaces
        symptoms = [s.strip() for s in re.split('[,\\s]+', text)]
        
        # Remove stopwords and non-word characters
        stop_words = set(stopwords.words('english'))
        symptoms = [re.sub(r'[^\w\s]', '', s) for s in symptoms]
        symptoms = [s for s in symptoms if s and s not in stop_words]
        
        return symptoms

    def predict_disease(self, symptoms_text):
        """Predict potential diseases with improved accuracy and confidence scoring"""
        try:
            # Preprocess input
            symptoms = self._preprocess_text(symptoms_text)
            
            if not symptoms:
                print("No valid symptoms found after preprocessing")
                return {}
            
            # Create weighted symptom text using severity
            weighted_symptoms = []
            matched_symptoms = []
            
            for symptom in symptoms:
                # Try to find the closest matching symptom
                best_match = None
                best_match_score = 0
                
                for known_symptom in self.unique_symptoms:
                    # Calculate similarity score
                    score = self._calculate_symptom_similarity(symptom, known_symptom)
                    if score > best_match_score and score > 0.5:
                        best_match = known_symptom
                        best_match_score = score
                
                if best_match:
                    matched_symptoms.append(best_match)
                    weight = self.symptom_severity_map.get(best_match, 1)
                    weighted_symptoms.extend([best_match] * weight)
            
            if not matched_symptoms:
                return {}
            
            # Get predictions using TF-IDF features
            try:
                text_features = self.vectorizer.transform([" ".join(weighted_symptoms)])
                
                # Get predictions from all models with adjusted thresholds
                svm_proba = self.svm.predict_proba(text_features)[0]
                gb_proba = self.gb.predict_proba(text_features)[0]
                ensemble_proba = self.ensemble.predict_proba(text_features)[0]
                
                # Weighted average of probabilities with adjusted weights
                probabilities = (0.3 * svm_proba + 0.7 * gb_proba + ensemble_proba) / 2
                
                # Get top 4 predictions with dynamic thresholds
                predictions = {}
                min_confidence = 0.10  # Lower minimum confidence threshold
                
                # Get indices of top 4 probabilities
                top_4_idx = np.argsort(probabilities)[-4:][::-1]
                
                # Add predictions with relative confidence scaling
                max_prob = probabilities[top_4_idx[0]]
                for idx in top_4_idx:
                    # Scale probabilities relative to the highest probability
                    scaled_prob = (probabilities[idx] / max_prob) * probabilities[idx]
                    if scaled_prob >= min_confidence:
                        predictions[self.disease_list[idx]] = float(scaled_prob)
                
                # Ensure at least 3 predictions
                if len(predictions) < 3:
                    for idx in top_4_idx:
                        if self.disease_list[idx] not in predictions:
                            predictions[self.disease_list[idx]] = float(probabilities[idx])
                            if len(predictions) >= 3:
                                break
                
                return dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True))
                
            except Exception as e:
                print(f"Error in prediction: {str(e)}")
                return {}
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return {}

    def _calculate_symptom_similarity(self, input_symptom, known_symptom):
        """Calculate similarity between input symptom and known symptom"""
        # Convert to lowercase and remove special characters
        input_symptom = re.sub(r'[^\w\s]', '', input_symptom.lower())
        known_symptom = re.sub(r'[^\w\s]', '', known_symptom.lower())
        
        # Exact match
        if input_symptom == known_symptom:
            return 1.0
        
        # Contains match
        if input_symptom in known_symptom or known_symptom in input_symptom:
            return 0.8
        
        # Calculate word-level similarity
        input_words = set(input_symptom.split())
        known_words = set(known_symptom.split())
        
        # Jaccard similarity
        intersection = len(input_words.intersection(known_words))
        union = len(input_words.union(known_words))
        
        if union == 0:
            return 0
        
        return intersection / union

    def get_detailed_recommendations(self, disease, probability):
        """Get detailed medical recommendations based on the predicted disease"""
        try:
            recommendations = []
            
            # Get disease information from dataset
            disease_info = self.dataset[self.dataset['Name'] == disease]
            if not disease_info.empty:
                # Add disease description and treatments
                treatments = disease_info['Treatments'].iloc[0]
                if isinstance(treatments, str):
                    recommendations.append(f"🏥 Treatments:")
                    for i, treatment in enumerate(treatments.split(','), 1):
                        if treatment.strip():
                            recommendations.append(f"  {i}. {treatment.strip()}")
                
                # Add symptoms with severity
                symptoms = disease_info['Symptoms'].iloc[0]
                if isinstance(symptoms, str):
                    symptoms_list = [s.strip() for s in symptoms.split(',')]
                    recommendations.append("\n🔍 Key Symptoms:")
                    for symptom in symptoms_list:
                        severity = self.severity_data[self.severity_data['Symptom'] == symptom]
                        if not severity.empty:
                            weight = severity['weight'].iloc[0]
                            if weight >= 7:
                                recommendations.append(f"  ⚠️ {symptom} (High Severity)")
                            elif weight >= 4:
                                recommendations.append(f"  ⚡ {symptom} (Moderate Severity)")
                            else:
                                recommendations.append(f"  • {symptom}")
            
            # Add confidence level
            if probability > 0.7:
                recommendations.insert(0, "🔴 High confidence prediction (>70%) - Consider seeking medical attention")
            elif probability > 0.4:
                recommendations.insert(0, "🟡 Moderate confidence prediction (40-70%) - Monitor symptoms closely")
            else:
                recommendations.insert(0, "🟢 Low confidence prediction (<40%) - Continue monitoring symptoms")
            
            # Add general recommendations
            recommendations.extend([
                "\n⚕️ General Advice:",
                "  • Consult with a healthcare professional for proper diagnosis",
                "  • Monitor your symptoms regularly",
                "  • Get adequate rest and stay hydrated",
                "  • Follow up if symptoms worsen"
            ])
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return [
                "Error retrieving specific recommendations",
                "Please consult with a healthcare professional",
                "Monitor your symptoms and seek medical attention if they worsen"
            ]

# Add missing np.nan_to_num fix at the beginning of the file
def np_nan_to_fill(array, value=0):
    """Helper function to replace NaN values in arrays"""
    return np.nan_to_num(array, nan=value)

        