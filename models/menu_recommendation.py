import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import joblib
import warnings
warnings.filterwarnings('ignore')

class MenuRecommendationModel:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.menu_similarity_matrix = None
        self.customer_preferences = None
        self.menu_df = None
        self.is_trained = False
        
    def prepare_menu_features(self, menu_df):
        """Prepare menu features for recommendation"""
        # Create feature text for each menu item
        menu_features = []
        
        for _, menu_item in menu_df.iterrows():
            # Combine category, ingredients, and mood tags
            ingredients = eval(menu_item['ingredients']) if isinstance(menu_item['ingredients'], str) else menu_item['ingredients']
            mood_tags = eval(menu_item['mood_tags']) if isinstance(menu_item['mood_tags'], str) else menu_item['mood_tags']
            
            feature_text = f"{menu_item['category']} {' '.join(ingredients)} {' '.join(mood_tags)}"
            menu_features.append(feature_text.lower())
        
        return menu_features
    
    def train_model(self, menu_df, orders_df, preferences_df=None):
        """Train the menu recommendation model"""
        print("Training menu recommendation model...")
        
        self.menu_df = menu_df.copy()
        
        # Prepare menu features
        menu_features = self.prepare_menu_features(menu_df)
        
        # Create TF-IDF vectors for menu items
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(menu_features)
        
        # Calculate similarity matrix
        self.menu_similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Create customer preferences if not provided
        if preferences_df is None:
            self.customer_preferences = self.create_customer_preferences(orders_df)
        else:
            self.customer_preferences = preferences_df
        
        self.is_trained = True
        print("Menu recommendation model trained successfully!")
        
        return self.menu_similarity_matrix
    
    def create_customer_preferences(self, orders_df):
        """Create customer preferences from order history"""
        customer_prefs = []
        
        # Get unique customers
        unique_customers = orders_df['customer_id'].unique()
        
        for customer_id in unique_customers:
            customer_orders = orders_df[orders_df['customer_id'] == customer_id]
            
            # Get favorite categories
            category_counts = customer_orders['category'].value_counts()
            favorite_categories = category_counts.head(3).index.tolist()
            
            # Get favorite menu items
            menu_counts = customer_orders['menu_name'].value_counts()
            favorite_menus = menu_counts.head(5).index.tolist()
            
            # Calculate average order value
            avg_order_value = customer_orders.groupby('order_id')['total_price'].sum().mean()
            
            # Get total visits
            total_visits = customer_orders['order_id'].nunique()
            
            customer_prefs.append({
                'customer_id': customer_id,
                'favorite_categories': ','.join(favorite_categories),
                'favorite_menus': ','.join(favorite_menus),
                'avg_order_value': avg_order_value,
                'total_visits': total_visits
            })
        
        return pd.DataFrame(customer_prefs)
    
    def get_content_based_recommendations(self, menu_name, top_n=5):
        """Get content-based recommendations based on menu similarity"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        # Find menu index
        menu_idx = self.menu_df[self.menu_df['name'] == menu_name].index
        if len(menu_idx) == 0:
            return pd.DataFrame()
        
        menu_idx = menu_idx[0]
        
        # Get similarity scores
        similarity_scores = list(enumerate(self.menu_similarity_matrix[menu_idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N similar menus (excluding the menu itself)
        similar_menus = similarity_scores[1:top_n+1]
        
        recommendations = []
        for idx, score in similar_menus:
            menu_item = self.menu_df.iloc[idx]
            recommendations.append({
                'menu_name': menu_item['name'],
                'category': menu_item['category'],
                'price': menu_item['price'],
                'similarity_score': score,
                'ingredients': menu_item['ingredients'],
                'mood_tags': menu_item['mood_tags']
            })
        
        return pd.DataFrame(recommendations)
    
    def get_collaborative_recommendations(self, customer_id, top_n=5):
        """Get collaborative filtering recommendations based on customer preferences"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        # Get customer preferences
        customer_pref = self.customer_preferences[self.customer_preferences['customer_id'] == customer_id]
        
        if len(customer_pref) == 0:
            return pd.DataFrame()
        
        customer_pref = customer_pref.iloc[0]
        
        # Get favorite categories and menus
        favorite_categories = customer_pref['favorite_categories'].split(',')
        favorite_menus = customer_pref['favorite_menus'].split(',')
        
        # Find menus in favorite categories
        recommended_menus = self.menu_df[self.menu_df['category'].isin(favorite_categories)]
        
        # Exclude already favorite menus
        recommended_menus = recommended_menus[~recommended_menus['name'].isin(favorite_menus)]
        
        # Sort by price (assuming customer prefers similar price range)
        avg_order_value = customer_pref['avg_order_value']
        recommended_menus['price_diff'] = abs(recommended_menus['price'] - avg_order_value)
        recommended_menus = recommended_menus.sort_values('price_diff').head(top_n)
        
        recommendations = []
        for _, menu_item in recommended_menus.iterrows():
            recommendations.append({
                'menu_name': menu_item['name'],
                'category': menu_item['category'],
                'price': menu_item['price'],
                'recommendation_reason': f"Based on your preference for {menu_item['category']}",
                'ingredients': menu_item['ingredients'],
                'mood_tags': menu_item['mood_tags']
            })
        
        return pd.DataFrame(recommendations)
    
    def get_mood_based_recommendations(self, mood, top_n=5):
        """Get recommendations based on mood"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        mood_mapping = {
            'comfort': ['comfort', 'traditional', 'warm'],
            'healthy': ['healthy', 'fresh'],
            'quick': ['quick'],
            'refreshing': ['refreshing'],
            'sweet': ['sweet'],
            'protein': ['protein'],
            'crispy': ['crispy']
        }
        
        target_moods = mood_mapping.get(mood.lower(), [mood.lower()])
        
        recommendations = []
        for _, menu_item in self.menu_df.iterrows():
            menu_moods = eval(menu_item['mood_tags']) if isinstance(menu_item['mood_tags'], str) else menu_item['mood_tags']
            
            # Check if any target mood matches menu moods
            if any(mood in menu_moods for mood in target_moods):
                recommendations.append({
                    'menu_name': menu_item['name'],
                    'category': menu_item['category'],
                    'price': menu_item['price'],
                    'mood_match': mood,
                    'ingredients': menu_item['ingredients'],
                    'mood_tags': menu_item['mood_tags']
                })
        
        recommendations_df = pd.DataFrame(recommendations)
        
        if len(recommendations_df) > top_n:
            recommendations_df = recommendations_df.head(top_n)
        
        return recommendations_df
    
    def get_ingredient_based_recommendations(self, ingredient, top_n=5):
        """Get recommendations based on ingredient"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        recommendations = []
        
        for _, menu_item in self.menu_df.iterrows():
            menu_ingredients = eval(menu_item['ingredients']) if isinstance(menu_item['ingredients'], str) else menu_item['ingredients']
            
            if ingredient.lower() in [ing.lower() for ing in menu_ingredients]:
                recommendations.append({
                    'menu_name': menu_item['name'],
                    'category': menu_item['category'],
                    'price': menu_item['price'],
                    'ingredient_match': ingredient,
                    'ingredients': menu_item['ingredients'],
                    'mood_tags': menu_item['mood_tags']
                })
        
        recommendations_df = pd.DataFrame(recommendations)
        
        if len(recommendations_df) > top_n:
            recommendations_df = recommendations_df.head(top_n)
        
        return recommendations_df
    
    def get_personalized_recommendations(self, customer_id, mood=None, ingredient=None, top_n=5):
        """Get personalized recommendations combining multiple factors"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        recommendations = []
        
        # Get collaborative recommendations
        collab_recs = self.get_collaborative_recommendations(customer_id, top_n)
        if len(collab_recs) > 0:
            for _, rec in collab_recs.iterrows():
                recommendations.append({
                    'menu_name': rec['menu_name'],
                    'category': rec['category'],
                    'price': rec['price'],
                    'recommendation_type': 'Collaborative',
                    'reason': rec['recommendation_reason'],
                    'ingredients': rec['ingredients'],
                    'mood_tags': rec['mood_tags']
                })
        
        # Add mood-based recommendations if specified
        if mood:
            mood_recs = self.get_mood_based_recommendations(mood, top_n)
            if len(mood_recs) > 0:
                for _, rec in mood_recs.iterrows():
                    recommendations.append({
                        'menu_name': rec['menu_name'],
                        'category': rec['category'],
                        'price': rec['price'],
                        'recommendation_type': 'Mood-based',
                        'reason': f"Perfect for {mood} mood",
                        'ingredients': rec['ingredients'],
                        'mood_tags': rec['mood_tags']
                    })
        
        # Add ingredient-based recommendations if specified
        if ingredient:
            ing_recs = self.get_ingredient_based_recommendations(ingredient, top_n)
            if len(ing_recs) > 0:
                for _, rec in ing_recs.iterrows():
                    recommendations.append({
                        'menu_name': rec['menu_name'],
                        'category': rec['category'],
                        'price': rec['price'],
                        'recommendation_type': 'Ingredient-based',
                        'reason': f"Contains {ingredient}",
                        'ingredients': rec['ingredients'],
                        'mood_tags': rec['mood_tags']
                    })
        
        # Remove duplicates and limit to top_n
        unique_recs = []
        seen_menus = set()
        
        for rec in recommendations:
            if rec['menu_name'] not in seen_menus:
                unique_recs.append(rec)
                seen_menus.add(rec['menu_name'])
            
            if len(unique_recs) >= top_n:
                break
        
        return pd.DataFrame(unique_recs)
    
    def save_model(self, filepath='models/menu_recommendation_model.pkl'):
        """Save the trained model"""
        if self.is_trained:
            model_data = {
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'menu_similarity_matrix': self.menu_similarity_matrix,
                'customer_preferences': self.customer_preferences,
                'menu_df': self.menu_df,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, filepath)
            print(f"Menu recommendation model saved to {filepath}")
        else:
            print("No trained model to save")
    
    def load_model(self, filepath='models/menu_recommendation_model.pkl'):
        """Load a trained model"""
        try:
            model_data = joblib.load(filepath)
            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.menu_similarity_matrix = model_data['menu_similarity_matrix']
            self.customer_preferences = model_data['customer_preferences']
            self.menu_df = model_data['menu_df']
            self.is_trained = model_data['is_trained']
            print(f"Menu recommendation model loaded from {filepath}")
        except FileNotFoundError:
            print(f"Model file {filepath} not found")
        except Exception as e:
            print(f"Error loading model: {e}")

def create_menu_recommendation_model(menu_df, orders_df, preferences_df=None):
    """Create and train menu recommendation model"""
    model = MenuRecommendationModel()
    similarity_matrix = model.train_model(menu_df, orders_df, preferences_df)
    model.save_model()
    return model, similarity_matrix 