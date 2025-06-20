import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
import joblib

class HDBPricePredictionModel:
    def __init__(self):
        self.gb_model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            random_state=42
        )
        self.nn_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        
    def preprocess_features(self, df):
        """Feature engineering and preprocessing"""
        df = df.copy()
        
        # Extract date features
        df['year'] = pd.to_datetime(df['month']).dt.year
        df['month_num'] = pd.to_datetime(df['month']).dt.month
        df['quarter'] = pd.to_datetime(df['month']).dt.quarter
        
        # Calculate remaining lease
        df['remaining_lease'] = 99 - (df['year'] - df['lease_commence_date'])
        
        # Extract storey information
        df['storey_min'] = df['storey_range'].str.extract('(\d+)').astype(int)
        df['storey_max'] = df['storey_range'].str.extract('TO (\d+)').fillna(
            df['storey_range'].str.extract('(\d+)')
        ).astype(int)
        df['storey_mid'] = (df['storey_min'] + df['storey_max']) / 2
        
        # Encode categorical variables
        categorical_features = ['town', 'flat_type', 'flat_model']
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                df[f'{feature}_encoded'] = self.label_encoders[feature].fit_transform(df[feature])
            else:
                df[f'{feature}_encoded'] = self.label_encoders[feature].transform(df[feature])
        
        # Price-related features
        df['price_per_sqm'] = df['resale_price'] / df['floor_area_sqm']
        
        return df
    
    def create_neural_network(self, input_dim):
        """Create deep neural network for price prediction"""
        model = keras.Sequential([
            keras.layers.Dense(512, activation='relu', input_dim=input_dim),
            keras.layers.Dropout(0.3),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def train(self, df):
        """Train ensemble model"""
        # Preprocess data
        df_processed = self.preprocess_features(df)
        
        # Select features
        feature_columns = [
            'floor_area_sqm', 'remaining_lease', 'storey_mid',
            'year', 'month_num', 'quarter',
            'town_encoded', 'flat_type_encoded', 'flat_model_encoded'
        ]
        
        X = df_processed[feature_columns]
        y = df_processed['resale_price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Gradient Boosting model
        self.gb_model.fit(X_train, y_train)
        
        # Train Neural Network
        self.nn_model = self.create_neural_network(X_train_scaled.shape[1])
        self.nn_model.fit(
            X_train_scaled, y_train,
            epochs=100,
            batch_size=64,
            validation_split=0.2,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ]
        )
        
        # Evaluate models
        gb_pred = self.gb_model.predict(X_test)
        nn_pred = self.nn_model.predict(X_test_scaled).flatten()
        
        # Ensemble prediction (weighted average)
        ensemble_pred = 0.6 * gb_pred + 0.4 * nn_pred
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, ensemble_pred)
        mse = mean_squared_error(y_test, ensemble_pred)
        r2 = r2_score(y_test, ensemble_pred)
        
        print(f"Model Performance:")
        print(f"MAE: ${mae:,.2f}")
        print(f"RMSE: ${np.sqrt(mse):,.2f}")
        print(f"R²: {r2:.4f}")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.gb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'mae': mae,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'feature_importance': self.feature_importance
        }
    
    def predict(self, X):
        """Make ensemble predictions"""
        X_scaled = self.scaler.transform(X)
        gb_pred = self.gb_model.predict(X)
        nn_pred = self.nn_model.predict(X_scaled).flatten()
        return 0.6 * gb_pred + 0.4 * nn_pred
    
    def predict_with_confidence(self, X, n_estimators=100):
        """Predict with confidence intervals using bootstrap"""
        predictions = []
        for _ in range(n_estimators):
            # Bootstrap sampling
            bootstrap_indices = np.random.choice(len(X), size=len(X), replace=True)
            X_bootstrap = X.iloc[bootstrap_indices]
            pred = self.predict(X_bootstrap)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        lower_ci = np.percentile(predictions, 2.5, axis=0)
        upper_ci = np.percentile(predictions, 97.5, axis=0)
        
        return mean_pred, lower_ci, upper_ci
    
    def save_model(self, path):
        """Save trained model"""
        joblib.dump({
            'gb_model': self.gb_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_importance': self.feature_importance
        }, f"{path}/hdb_price_model.pkl")
        
        self.nn_model.save(f"{path}/nn_model.h5")
    
    def load_model(self, path):
        """Load trained model"""
        model_data = joblib.load(f"{path}/hdb_price_model.pkl")
        self.gb_model = model_data['gb_model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_importance = model_data['feature_importance']
        self.nn_model = keras.models.load_model(f"{path}/nn_model.h5")