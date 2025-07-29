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

class InventoryManagementModel:
    def __init__(self):
        self.ingredient_models = {}
        self.scalers = {}
        self.inventory_df = None
        self.orders_df = None
        self.menu_df = None
        self.is_trained = False
        
    def prepare_ingredient_usage_data(self, orders_df, menu_df):
        """Prepare ingredient usage data from orders and menu"""
        # Create ingredient usage mapping
        ingredient_usage = []
        
        for _, menu_item in menu_df.iterrows():
            menu_name = menu_item['name']
            ingredients = eval(menu_item['ingredients']) if isinstance(menu_item['ingredients'], str) else menu_item['ingredients']
            
            # Get orders for this menu
            menu_orders = orders_df[orders_df['menu_name'] == menu_name]
            
            for _, order in menu_orders.iterrows():
                # Estimate ingredient usage (simplified - in real scenario would have actual recipes)
                for ingredient in ingredients:
                    # Assume 1 unit of ingredient per menu item (simplified)
                    usage_amount = order['quantity']
                    
                    ingredient_usage.append({
                        'date': order['order_date'],
                        'ingredient': ingredient.strip("'"),
                        'usage_amount': usage_amount,
                        'menu_name': menu_name,
                        'day_of_week': order['day_of_week'],
                        'is_weekend': order['is_weekend']
                    })
        
        return pd.DataFrame(ingredient_usage)
    
    def train_model(self, orders_df, menu_df, inventory_df):
        """Train inventory management models for each ingredient"""
        print("Training inventory management models...")
        
        self.orders_df = orders_df.copy()
        self.menu_df = menu_df.copy()
        self.inventory_df = inventory_df.copy()
        
        # Prepare ingredient usage data
        usage_data = self.prepare_ingredient_usage_data(orders_df, menu_df)
        
        # Train model for each ingredient
        unique_ingredients = usage_data['ingredient'].unique()
        
        for ingredient in unique_ingredients:
            print(f"Training model for {ingredient}...")
            
            # Get usage data for this ingredient
            ingredient_data = usage_data[usage_data['ingredient'] == ingredient].copy()
            
            if len(ingredient_data) < 10:  # Need minimum data points
                continue
            
            # Aggregate daily usage
            daily_usage = ingredient_data.groupby('date').agg({
                'usage_amount': 'sum'
            }).reset_index()
            
            # Add time features
            daily_usage['date'] = pd.to_datetime(daily_usage['date'])
            daily_usage['day_of_week'] = daily_usage['date'].dt.dayofweek
            daily_usage['month'] = daily_usage['date'].dt.month
            daily_usage['day_of_month'] = daily_usage['date'].dt.day
            daily_usage['is_weekend'] = daily_usage['day_of_week'].isin([5, 6]).astype(int)
            
            # Create lag features
            daily_usage['usage_lag1'] = daily_usage['usage_amount'].shift(1)
            daily_usage['usage_lag7'] = daily_usage['usage_amount'].shift(7)
            daily_usage['usage_ma7'] = daily_usage['usage_amount'].rolling(window=7).mean()
            daily_usage['usage_ma30'] = daily_usage['usage_amount'].rolling(window=30).mean()
            
            # Remove NaN values
            daily_usage = daily_usage.dropna()
            
            if len(daily_usage) < 5:
                continue
            
            # Prepare features and target
            feature_columns = ['day_of_week', 'month', 'day_of_month', 'is_weekend',
                              'usage_lag1', 'usage_lag7', 'usage_ma7', 'usage_ma30']
            
            X = daily_usage[feature_columns]
            y = daily_usage['usage_amount']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model (use Random Forest for better performance)
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store model and scaler
            self.ingredient_models[ingredient] = {
                'model': model,
                'scaler': scaler,
                'feature_columns': feature_columns,
                'mse': mse,
                'r2': r2
            }
            
            print(f"  {ingredient}: MSE={mse:.2f}, R²={r2:.3f}")
        
        self.is_trained = True
        print(f"Trained models for {len(self.ingredient_models)} ingredients")
        
        return len(self.ingredient_models)
    
    def predict_ingredient_usage(self, ingredient, days_ahead=7):
        """Predict ingredient usage for the next N days"""
        if not self.is_trained or ingredient not in self.ingredient_models:
            return None
        
        model_info = self.ingredient_models[ingredient]
        model = model_info['model']
        scaler = model_info['scaler']
        feature_columns = model_info['feature_columns']
        
        # Get historical usage data for this ingredient
        usage_data = self.prepare_ingredient_usage_data(self.orders_df, self.menu_df)
        ingredient_data = usage_data[usage_data['ingredient'] == ingredient]
        
        if len(ingredient_data) == 0:
            return None
        
        # Aggregate daily usage
        daily_usage = ingredient_data.groupby('date').agg({
            'usage_amount': 'sum'
        }).reset_index()
        
        daily_usage['date'] = pd.to_datetime(daily_usage['date'])
        daily_usage['day_of_week'] = daily_usage['date'].dt.dayofweek
        daily_usage['month'] = daily_usage['date'].dt.month
        daily_usage['day_of_month'] = daily_usage['date'].dt.day
        daily_usage['is_weekend'] = daily_usage['day_of_week'].isin([5, 6]).astype(int)
        daily_usage['usage_lag1'] = daily_usage['usage_amount'].shift(1)
        daily_usage['usage_lag7'] = daily_usage['usage_amount'].shift(7)
        daily_usage['usage_ma7'] = daily_usage['usage_amount'].rolling(window=7).mean()
        daily_usage['usage_ma30'] = daily_usage['usage_amount'].rolling(window=30).mean()
        
        # Get last values for prediction
        last_usage = daily_usage['usage_amount'].iloc[-1] if len(daily_usage) > 0 else 0
        last_usage_7 = daily_usage['usage_amount'].iloc[-7] if len(daily_usage) >= 7 else 0
        last_ma7 = daily_usage['usage_amount'].tail(7).mean() if len(daily_usage) >= 7 else 0
        last_ma30 = daily_usage['usage_amount'].tail(30).mean() if len(daily_usage) >= 30 else 0
        
        predictions = []
        last_date = datetime.now().date()
        
        for i in range(1, days_ahead + 1):
            future_date = last_date + timedelta(days=i)
            
            # Create features for future date
            features = {
                'day_of_week': future_date.weekday(),
                'month': future_date.month,
                'day_of_month': future_date.day,
                'is_weekend': 1 if future_date.weekday() >= 5 else 0,
                'usage_lag1': last_usage,
                'usage_lag7': last_usage_7,
                'usage_ma7': last_ma7,
                'usage_ma30': last_ma30
            }
            
            # Convert to DataFrame and scale
            features_df = pd.DataFrame([features])
            features_scaled = scaler.transform(features_df)
            
            # Make prediction
            predicted_usage = model.predict(features_scaled)[0]
            
            predictions.append({
                'date': future_date,
                'ingredient': ingredient,
                'predicted_usage': max(0, round(predicted_usage, 2)),
                'day_of_week': future_date.strftime('%A')
            })
        
        return pd.DataFrame(predictions)
    
    def get_inventory_alerts(self, inventory_df, days_ahead=7):
        """Get inventory alerts based on predicted usage"""
        alerts = []
        
        for _, inventory_item in inventory_df.iterrows():
            ingredient = inventory_item['ingredient_name']
            current_stock = inventory_item['current_stock']
            reorder_point = inventory_item['reorder_point']
            
            # Predict usage for this ingredient
            usage_prediction = self.predict_ingredient_usage(ingredient, days_ahead)
            
            if usage_prediction is not None:
                total_predicted_usage = usage_prediction['predicted_usage'].sum()
                
                # Check if stock will be insufficient
                if current_stock - total_predicted_usage <= reorder_point:
                    alerts.append({
                        'ingredient': ingredient,
                        'current_stock': current_stock,
                        'predicted_usage': total_predicted_usage,
                        'remaining_stock': current_stock - total_predicted_usage,
                        'reorder_point': reorder_point,
                        'alert_type': 'Low Stock Warning',
                        'recommended_order': max(0, reorder_point - (current_stock - total_predicted_usage)),
                        'unit': inventory_item['unit']
                    })
        
        return pd.DataFrame(alerts)
    
    def optimize_reorder_points(self, inventory_df, confidence_level=0.95):
        """Optimize reorder points based on usage patterns"""
        optimized_inventory = inventory_df.copy()
        
        for _, inventory_item in inventory_df.iterrows():
            ingredient = inventory_item['ingredient_name']
            
            if ingredient in self.ingredient_models:
                # Get historical usage data
                usage_data = self.prepare_ingredient_usage_data(self.orders_df, self.menu_df)
                ingredient_data = usage_data[usage_data['ingredient'] == ingredient]
                
                if len(ingredient_data) > 0:
                    # Calculate usage statistics
                    daily_usage = ingredient_data.groupby('date')['usage_amount'].sum()
                    
                    # Calculate optimal reorder point based on usage variability
                    mean_usage = daily_usage.mean()
                    std_usage = daily_usage.std()
                    
                    # Safety stock based on confidence level
                    safety_stock = std_usage * 1.96  # 95% confidence interval
                    
                    # Optimal reorder point = lead time demand + safety stock
                    # Assuming 3-day lead time
                    lead_time_days = 3
                    lead_time_demand = mean_usage * lead_time_days
                    optimal_reorder_point = lead_time_demand + safety_stock
                    
                    # Update reorder point
                    optimized_inventory.loc[
                        optimized_inventory['ingredient_name'] == ingredient, 
                        'reorder_point'
                    ] = max(1, round(optimal_reorder_point))
        
        return optimized_inventory
    
    def get_inventory_analytics(self, inventory_df):
        """Get comprehensive inventory analytics"""
        analytics = {}
        
        # Basic statistics
        analytics['total_items'] = len(inventory_df)
        analytics['low_stock_items'] = len(inventory_df[inventory_df['status'] == 'Low Stock'])
        analytics['low_stock_percentage'] = (analytics['low_stock_items'] / analytics['total_items']) * 100
        
        # Value analysis
        analytics['total_inventory_value'] = inventory_df['current_stock'].sum()
        
        # Category analysis
        if 'category' in inventory_df.columns:
            category_stats = inventory_df.groupby('category').agg({
                'current_stock': ['count', 'sum'],
                'status': lambda x: (x == 'Low Stock').sum()
            }).round(2)
            analytics['category_analysis'] = category_stats
        
        # Usage prediction for next 7 days
        if self.is_trained:
            total_predicted_usage = 0
            for ingredient in inventory_df['ingredient_name']:
                usage_pred = self.predict_ingredient_usage(ingredient, 7)
                if usage_pred is not None:
                    total_predicted_usage += usage_pred['predicted_usage'].sum()
            
            analytics['predicted_usage_7_days'] = total_predicted_usage
        
        return analytics
    
    def generate_inventory_report(self, inventory_df, days_ahead=7):
        """Generate comprehensive inventory report"""
        report = {
            'summary': {},
            'alerts': [],
            'recommendations': [],
            'predictions': {}
        }
        
        # Summary statistics
        report['summary'] = {
            'total_items': len(inventory_df),
            'low_stock_items': len(inventory_df[inventory_df['status'] == 'Low Stock']),
            'normal_stock_items': len(inventory_df[inventory_df['status'] == 'Normal']),
            'total_inventory_value': inventory_df['current_stock'].sum()
        }
        
        # Get alerts
        alerts_df = self.get_inventory_alerts(inventory_df, days_ahead)
        report['alerts'] = alerts_df.to_dict('records') if len(alerts_df) > 0 else []
        
        # Generate recommendations
        for _, alert in alerts_df.iterrows():
            report['recommendations'].append({
                'ingredient': alert['ingredient'],
                'action': 'Order',
                'quantity': alert['recommended_order'],
                'unit': alert['unit'],
                'reason': f"Stock will be below reorder point in {days_ahead} days"
            })
        
        # Get predictions for all ingredients
        for _, inventory_item in inventory_df.iterrows():
            ingredient = inventory_item['ingredient_name']
            usage_pred = self.predict_ingredient_usage(ingredient, days_ahead)
            
            if usage_pred is not None:
                report['predictions'][ingredient] = {
                    'total_predicted_usage': usage_pred['predicted_usage'].sum(),
                    'daily_predictions': usage_pred.to_dict('records')
                }
        
        return report
    
    def save_model(self, filepath='models/inventory_management_model.pkl'):
        """Save the trained model"""
        if self.is_trained:
            model_data = {
                'ingredient_models': self.ingredient_models,
                'inventory_df': self.inventory_df,
                'orders_df': self.orders_df,
                'menu_df': self.menu_df,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, filepath)
            print(f"Inventory management model saved to {filepath}")
        else:
            print("No trained model to save")
    
    def load_model(self, filepath='models/inventory_management_model.pkl'):
        """Load a trained model"""
        try:
            model_data = joblib.load(filepath)
            self.ingredient_models = model_data['ingredient_models']
            self.inventory_df = model_data['inventory_df']
            self.orders_df = model_data['orders_df']
            self.menu_df = model_data['menu_df']
            self.is_trained = model_data['is_trained']
            print(f"Inventory management model loaded from {filepath}")
        except FileNotFoundError:
            print(f"Model file {filepath} not found")
        except Exception as e:
            print(f"Error loading model: {e}")

def create_inventory_management_model(orders_df, menu_df, inventory_df):
    """Create and train inventory management model"""
    model = InventoryManagementModel()
    num_models = model.train_model(orders_df, menu_df, inventory_df)
    model.save_model()
    return model, num_models 