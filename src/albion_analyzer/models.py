"""
Machine Learning Models for Market Manipulation Detection
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, List, Optional
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class ManipulationDetector:
    """
    Ensemble machine learning system for detecting market manipulation in gaming economies.
    
    Combines multiple anomaly detection algorithms (Isolation Forest, Local Outlier Factor,
    and One-Class SVM) using ensemble voting to provide robust manipulation detection with
    high confidence scoring. Automatically handles feature engineering and scaling for
    optimal performance across different market conditions.
    
    The ensemble approach reduces false positives by requiring agreement between multiple
    algorithms, while confidence scoring helps prioritize the most suspicious cases for
    investigation. Supports both supervised evaluation (with ground truth) and unsupervised
    detection for real-world deployment.
    
    Attributes:
        contamination (float): Expected proportion of manipulation cases in dataset
        models (dict): Dictionary of trained ML models  
        scaler (StandardScaler): Feature scaling transformer
        is_fitted (bool): Whether models have been trained
    """
    
    def __init__(self, contamination: float = 0.01) -> None:
        self.contamination = contamination
        self.models = {
            'isolation_forest': IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=200
            ),
            'lof': LocalOutlierFactor(
                contamination=contamination,
                n_neighbors=20,
                novelty=True
            ),
            'one_class_svm': OneClassSVM(
                gamma='scale',
                nu=contamination
            )
        }
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def prepare_features(self, df):
        """
        Extract features for manipulation detection
        """
        features = []
        
        # Rolling statistics features
        for window in [3, 7, 14]:
            df[f'price_rolling_mean_{window}'] = df.groupby(['item', 'city'])['price'].transform(
                lambda x: x.rolling(window, min_periods=2).mean()
            )
            df[f'price_rolling_std_{window}'] = df.groupby(['item', 'city'])['price'].transform(
                lambda x: x.rolling(window, min_periods=2).std()
            )
        
        # Existing features
        if 'z' in df.columns:
            features.extend(['z', 'peer_dev'])
        
        # Rolling mean features
        rolling_features = [col for col in df.columns if 'rolling' in col]
        features.extend(rolling_features)
        
        # Quality-based features (if available)
        if 'quality' in df.columns:
            features.append('quality')
        
        # Bid-ask spread features (if available)
        spread_features = [col for col in df.columns if 'spread' in col]
        features.extend(spread_features)
        
        # Volume features (if available)
        if 'volume' in df.columns:
            features.append('volume')
        
        # Filter to only existing columns
        available_features = [f for f in features if f in df.columns]
        
        if not available_features:
            raise ValueError("No suitable features found in DataFrame")
        
        return df[available_features].fillna(0)
    
    def fit(self, df):
        """
        Train all models on the dataset
        """
        print("Preparing features for training...")
        X = self.prepare_features(df)
        print(f"Using features: {list(X.columns)}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        print("Training models...")
        for name, model in self.models.items():
            print(f"  Training {name}...")
            if name == 'lof':
                # LOF requires fit_predict for training
                model.fit(X_scaled)
            else:
                model.fit(X_scaled)
        
        self.is_fitted = True
        print("Training complete!")
        
    def predict(self, df: pd.DataFrame) -> Dict[str, Union[pd.Series, np.ndarray, Dict]]:
        """
        Predict market manipulation using ensemble voting across multiple ML algorithms.
        
        Applies all trained models to the input data and combines their predictions using
        majority voting. Cases are flagged as manipulation if at least 2 out of 3 models
        agree. Confidence scores are calculated by averaging normalized decision scores
        from all models to provide interpretable manipulation likelihood.
        
        Args:
            df: DataFrame containing market data with same features used in training.
                Must include all features that were present during fit() call.
                
        Returns:
            Dictionary containing prediction results:
            - manipulation_detected (pd.Series): Boolean series indicating manipulation cases
            - confidence_score (np.ndarray): Confidence scores (0-1) for each prediction
            - individual_predictions (dict): Raw predictions from each model
            - individual_scores (dict): Raw decision scores from each model
            
        Raises:
            ValueError: If models have not been fitted or if input features don't match training data.
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction")
        
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        scores = {}
        
        for name, model in self.models.items():
            if name == 'lof':
                pred = model.predict(X_scaled)
                score = model.negative_outlier_factor_
            elif name == 'isolation_forest':
                pred = model.predict(X_scaled)
                score = model.decision_function(X_scaled)
            else:  # one_class_svm
                pred = model.predict(X_scaled)
                score = model.decision_function(X_scaled)
            
            predictions[name] = pred
            scores[name] = score
        
        # Ensemble voting (majority rule)
        pred_df = pd.DataFrame(predictions)
        ensemble_pred = (pred_df == -1).sum(axis=1) >= 2  # At least 2 models agree
        
        # Confidence score (average of normalized scores)
        confidence_scores = []
        for name, score in scores.items():
            # Normalize scores to 0-1 range
            norm_score = (score - score.min()) / (score.max() - score.min() + 1e-8)
            confidence_scores.append(norm_score)
        
        avg_confidence = np.mean(confidence_scores, axis=0)
        
        return {
            'manipulation_detected': ensemble_pred,
            'confidence_score': avg_confidence,
            'individual_predictions': predictions,
            'individual_scores': scores
        }
    
    def evaluate_model_performance(self, df, true_labels=None):
        """
        Evaluate model performance if ground truth is available
        """
        if true_labels is None:
            print("No ground truth labels provided - showing prediction statistics only")
            results = self.predict(df)
            manipulation_rate = results['manipulation_detected'].mean()
            avg_confidence = results['confidence_score'].mean()
            
            print(f"Manipulation detection rate: {manipulation_rate:.1%}")
            print(f"Average confidence score: {avg_confidence:.3f}")
            return results
        
        results = self.predict(df)
        pred_labels = results['manipulation_detected']
        
        print("Model Performance:")
        print(classification_report(true_labels, pred_labels))
        print("\nConfusion Matrix:")
        print(confusion_matrix(true_labels, pred_labels))
        
        return results

class SimpleRuleBasedDetector:
    """
    Simple rule-based detector for comparison and validation
    """
    
    def __init__(self, z_threshold=2.0, peer_dev_threshold=1.5, spread_threshold=50.0):
        self.z_threshold = z_threshold
        self.peer_dev_threshold = peer_dev_threshold
        self.spread_threshold = spread_threshold
    
    def predict(self, df):
        """
        Predict manipulation using simple rules
        """
        manipulation_detected = pd.Series(False, index=df.index)
        confidence_score = pd.Series(0.0, index=df.index)
        
        # Rule 1: High z-score deviation
        if 'z' in df.columns:
            high_z = df['z'].abs() > self.z_threshold
            manipulation_detected |= high_z
            confidence_score += high_z * 0.3
        
        # Rule 2: High peer deviation
        if 'peer_dev' in df.columns:
            high_peer_dev = df['peer_dev'].abs() > self.peer_dev_threshold
            manipulation_detected |= high_peer_dev
            confidence_score += high_peer_dev * 0.4
        
        # Rule 3: High bid-ask spread
        spread_cols = [col for col in df.columns if 'spread_pct' in col]
        if spread_cols:
            high_spread = df[spread_cols[0]] > self.spread_threshold
            manipulation_detected |= high_spread
            confidence_score += high_spread * 0.3
        
        return {
            'manipulation_detected': manipulation_detected,
            'confidence_score': confidence_score.clip(0, 1)
        }

def compare_models(df, true_labels=None):
    """
    Compare different detection approaches
    """
    print("=== Model Comparison ===")
    
    # ML Ensemble
    print("\n1. ML Ensemble Detector:")
    ml_detector = ManipulationDetector()
    ml_detector.fit(df)
    ml_results = ml_detector.evaluate_model_performance(df, true_labels)
    
    # Rule-based
    print("\n2. Rule-Based Detector:")
    rule_detector = SimpleRuleBasedDetector()
    rule_results = rule_detector.predict(df)
    rule_manipulation_rate = rule_results['manipulation_detected'].mean()
    print(f"Manipulation detection rate: {rule_manipulation_rate:.1%}")
    
    if true_labels is not None:
        print(classification_report(true_labels, rule_results['manipulation_detected']))
    
    return {
        'ml_results': ml_results,
        'rule_results': rule_results
    }