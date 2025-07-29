import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Page configuration
st.set_page_config(
    page_title="Sistem Manajemen Restoran AI",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def generate_sample_data():
    """Generate simple sample data"""
    # Sample menu data
    menu_data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Nasi Goreng', 'Mie Goreng', 'Ayam Goreng', 'Sate Ayam', 'Gado-gado'],
        'category': ['Main Course', 'Main Course', 'Main Course', 'Main Course', 'Appetizer'],
        'price': [25000, 22000, 30000, 28000, 18000]
    }
    
    # Sample orders data
    orders_data = {
        'order_id': range(1, 101),
        'menu_id': np.random.randint(1, 6, 100),
        'quantity': np.random.randint(1, 4, 100),
        'order_date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
        'total_price': np.random.randint(15000, 50000, 100)
    }
    
    # Sample inventory data
    inventory_data = {
        'ingredient_name': ['Nasi', 'Mie', 'Ayam', 'Telur', 'Sayuran'],
        'current_stock': [50, 30, 40, 100, 20],
        'reorder_point': [10, 5, 8, 20, 5],
        'status': ['Normal', 'Normal', 'Normal', 'Normal', 'Low Stock']
    }
    
    return pd.DataFrame(menu_data), pd.DataFrame(orders_data), pd.DataFrame(inventory_data)

def show_dashboard(menu_df, orders_df, inventory_df):
    """Show dashboard with metrics"""
    st.header("🏠 Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Orders", len(orders_df))
    
    with col2:
        st.metric("Total Revenue", f"Rp {orders_df['total_price'].sum():,}")
    
    with col3:
        st.metric("Menu Items", len(menu_df))
    
    with col4:
        low_stock = len(inventory_df[inventory_df['status'] == 'Low Stock'])
        st.metric("Low Stock Items", low_stock)
    
    # Charts
    st.subheader("📊 Analytics")
    
    # Revenue trend
    daily_revenue = orders_df.groupby('order_date')['total_price'].sum().reset_index()
    st.line_chart(daily_revenue.set_index('order_date'))
    
    # Menu popularity
    menu_popularity = orders_df.groupby('menu_id')['quantity'].sum().reset_index()
    menu_popularity = menu_popularity.merge(menu_df, left_on='menu_id', right_on='id')
    st.bar_chart(menu_popularity.set_index('name')['quantity'])

def show_menu_recommendations(menu_df, orders_df):
    """Show menu recommendations"""
    st.header("🍽️ Rekomendasi Menu")
    
    # Simple recommendations based on popularity
    menu_popularity = orders_df.groupby('menu_id')['quantity'].sum().reset_index()
    menu_popularity = menu_popularity.merge(menu_df, left_on='menu_id', right_on='id')
    menu_popularity = menu_popularity.sort_values('quantity', ascending=False)
    
    st.subheader("Menu Terpopuler")
    for _, row in menu_popularity.head(5).iterrows():
        st.write(f"🍽️ **{row['name']}** - Rp {row['price']:,}")

def show_inventory_management(inventory_df):
    """Show inventory management"""
    st.header("📦 Pengelolaan Inventaris")
    
    # Show inventory status
    st.subheader("Status Inventaris")
    st.dataframe(inventory_df)
    
    # Low stock alerts
    low_stock_items = inventory_df[inventory_df['status'] == 'Low Stock']
    if len(low_stock_items) > 0:
        st.warning(f"⚠️ Ada {len(low_stock_items)} item dengan stok rendah!")
        st.dataframe(low_stock_items)

def main():
    # Header
    st.markdown('<h1 class="main-header">🍽️ Sistem Manajemen Restoran AI</h1>', unsafe_allow_html=True)
    
    # Generate sample data
    menu_df, orders_df, inventory_df = generate_sample_data()
    
    # Sidebar navigation
    st.sidebar.title("📊 Menu Navigasi")
    page = st.sidebar.selectbox(
        "Pilih Halaman:",
        ["🏠 Dashboard", "🍽️ Rekomendasi Menu", "📦 Pengelolaan Inventaris"]
    )
    
    # Display selected page
    if page == "🏠 Dashboard":
        show_dashboard(menu_df, orders_df, inventory_df)
    elif page == "🍽️ Rekomendasi Menu":
        show_menu_recommendations(menu_df, orders_df)
    elif page == "📦 Pengelolaan Inventaris":
        show_inventory_management(inventory_df)

if __name__ == "__main__":
    main() 