import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DemandForecastModel:
    def __init__(self):
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.lr_model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, orders_df):
        """Prepare features for demand forecasting"""
        # Aggregate daily data
        daily_data = orders_df.groupby('order_date').agg({
            'quantity': 'sum',
            'total_price': 'sum',
            'order_id': 'count'
        }).reset_index()
        
        daily_data.columns = ['date', 'total_quantity', 'total_revenue', 'num_orders']
        
        # Extract time-based features
        daily_data['day_of_week'] = daily_data['date'].dt.dayofweek
        daily_data['month'] = daily_data['date'].dt.month
        daily_data['day_of_month'] = daily_data['date'].dt.day
        daily_data['is_weekend'] = daily_data['day_of_week'].isin([5, 6]).astype(int)
        
        # Create lag features
        daily_data['quantity_lag1'] = daily_data['total_quantity'].shift(1)
        daily_data['quantity_lag7'] = daily_data['total_quantity'].shift(7)
        daily_data['revenue_lag1'] = daily_data['total_revenue'].shift(1)
        
        # Create rolling averages
        daily_data['quantity_ma7'] = daily_data['total_quantity'].rolling(window=7).mean()
        daily_data['quantity_ma30'] = daily_data['total_quantity'].rolling(window=30).mean()
        
        # Remove NaN values
        daily_data = daily_data.dropna()
        
        return daily_data
    
    def train_model(self, orders_df):
        """Train the demand forecasting model"""
        print("Preparing features for demand forecasting...")
        daily_data = self.prepare_features(orders_df)
        
        # Define features and target
        feature_columns = ['day_of_week', 'month', 'day_of_month', 'is_weekend',
                          'quantity_lag1', 'quantity_lag7', 'revenue_lag1',
                          'quantity_ma7', 'quantity_ma30']
        
        X = daily_data[feature_columns]
        y_quantity = daily_data['total_quantity']
        y_revenue = daily_data['total_revenue']
        
        # Split data
        X_train, X_test, y_train_q, y_test_q = train_test_split(
            X, y_quantity, test_size=0.2, random_state=42
        )
        _, _, y_train_r, y_test_r = train_test_split(
            X, y_revenue, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest for quantity
        print("Training Random Forest model for quantity prediction...")
        self.rf_model.fit(X_train_scaled, y_train_q)
        
        # Train Linear Regression for revenue
        print("Training Linear Regression model for revenue prediction...")
        self.lr_model.fit(X_train_scaled, y_train_r)
        
        # Evaluate models
        rf_pred = self.rf_model.predict(X_test_scaled)
        lr_pred = self.lr_model.predict(X_test_scaled)
        
        rf_mse = mean_squared_error(y_test_q, rf_pred)
        rf_r2 = r2_score(y_test_q, rf_pred)
        
        lr_mse = mean_squared_error(y_test_r, lr_pred)
        lr_r2 = r2_score(y_test_r, lr_pred)
        
        print(f"Random Forest (Quantity) - MSE: {rf_mse:.2f}, R²: {rf_r2:.3f}")
        print(f"Linear Regression (Revenue) - MSE: {lr_mse:.2f}, R²: {lr_r2:.3f}")
        
        self.is_trained = True
        self.feature_columns = feature_columns
        
        return {
            'rf_mse': rf_mse,
            'rf_r2': rf_r2,
            'lr_mse': lr_mse,
            'lr_r2': lr_r2
        }
    
    def predict_demand(self, days_ahead=7):
        """Predict demand for the next N days"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get the last available data point
        last_date = datetime.now().date()
        
        predictions = []
        
        for i in range(1, days_ahead + 1):
            future_date = last_date + timedelta(days=i)
            
            # Create features for future date
            features = {
                'day_of_week': future_date.weekday(),
                'month': future_date.month,
                'day_of_month': future_date.day,
                'is_weekend': 1 if future_date.weekday() >= 5 else 0,
                'quantity_lag1': 0,  # Will be updated with actual predictions
                'quantity_lag7': 0,  # Will be updated with actual predictions
                'revenue_lag1': 0,   # Will be updated with actual predictions
                'quantity_ma7': 0,   # Will be updated with actual predictions
                'quantity_ma30': 0   # Will be updated with actual predictions
            }
            
            # Convert to DataFrame
            features_df = pd.DataFrame([features])
            features_scaled = self.scaler.transform(features_df)
            
            # Make predictions
            quantity_pred = self.rf_model.predict(features_scaled)[0]
            revenue_pred = self.lr_model.predict(features_scaled)[0]
            
            predictions.append({
                'date': future_date,
                'predicted_quantity': max(0, round(quantity_pred)),
                'predicted_revenue': max(0, round(revenue_pred)),
                'day_of_week': future_date.strftime('%A')
            })
        
        return pd.DataFrame(predictions)
    
    def predict_menu_demand(self, orders_df, menu_name, days_ahead=7):
        """Predict demand for specific menu items"""
        # Filter orders for specific menu
        menu_orders = orders_df[orders_df['menu_name'] == menu_name].copy()
        
        if len(menu_orders) == 0:
            return None
        
        # Aggregate daily menu orders
        daily_menu_data = menu_orders.groupby('order_date').agg({
            'quantity': 'sum'
        }).reset_index()
        
        # Prepare features for menu-specific prediction
        daily_menu_data['day_of_week'] = daily_menu_data['order_date'].dt.dayofweek
        daily_menu_data['month'] = daily_menu_data['order_date'].dt.month
        daily_menu_data['day_of_month'] = daily_menu_data['order_date'].dt.day
        daily_menu_data['is_weekend'] = daily_menu_data['day_of_week'].isin([5, 6]).astype(int)
        
        # Create lag features
        daily_menu_data['quantity_lag1'] = daily_menu_data['quantity'].shift(1)
        daily_menu_data['quantity_lag7'] = daily_menu_data['quantity'].shift(7)
        daily_menu_data['quantity_ma7'] = daily_menu_data['quantity'].rolling(window=7).mean()
        
        # Remove NaN values
        daily_menu_data = daily_menu_data.dropna()
        
        if len(daily_menu_data) < 10:
            return None
        
        # Train a simple model for this menu
        feature_cols = ['day_of_week', 'month', 'day_of_month', 'is_weekend',
                       'quantity_lag1', 'quantity_lag7', 'quantity_ma7']
        
        X = daily_menu_data[feature_cols]
        y = daily_menu_data['quantity']
        
        # Use simple linear regression for menu-specific prediction
        menu_model = LinearRegression()
        menu_model.fit(X, y)
        
        # Make predictions
        predictions = []
        last_date = datetime.now().date()
        
        for i in range(1, days_ahead + 1):
            future_date = last_date + timedelta(days=i)
            
            features = {
                'day_of_week': future_date.weekday(),
                'month': future_date.month,
                'day_of_month': future_date.day,
                'is_weekend': 1 if future_date.weekday() >= 5 else 0,
                'quantity_lag1': daily_menu_data['quantity'].iloc[-1] if len(daily_menu_data) > 0 else 0,
                'quantity_lag7': daily_menu_data['quantity'].iloc[-7] if len(daily_menu_data) >= 7 else 0,
                'quantity_ma7': daily_menu_data['quantity'].tail(7).mean() if len(daily_menu_data) >= 7 else 0
            }
            
            features_df = pd.DataFrame([features])
            pred_quantity = menu_model.predict(features_df)[0]
            
            predictions.append({
                'date': future_date,
                'menu_name': menu_name,
                'predicted_quantity': max(0, round(pred_quantity)),
                'day_of_week': future_date.strftime('%A')
            })
        
        return pd.DataFrame(predictions)
    
    def save_model(self, filepath='models/demand_forecast_model.pkl'):
        """Save the trained model"""
        if self.is_trained:
            model_data = {
                'rf_model': self.rf_model,
                'lr_model': self.lr_model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No trained model to save")
    
    def load_model(self, filepath='models/demand_forecast_model.pkl'):
        """Load a trained model"""
        try:
            model_data = joblib.load(filepath)
            self.rf_model = model_data['rf_model']
            self.lr_model = model_data['lr_model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data['is_trained']
            print(f"Model loaded from {filepath}")
        except FileNotFoundError:
            print(f"Model file {filepath} not found")
        except Exception as e:
            print(f"Error loading model: {e}")

def create_demand_forecast_model(orders_df):
    """Create and train demand forecast model"""
    model = DemandForecastModel()
    metrics = model.train_model(orders_df)
    model.save_model()
    return model, metrics 