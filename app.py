import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from utils.data_generator import DataGenerator
from utils.helpers import (
    load_data, get_low_stock_alerts, calculate_metrics, format_currency,
    get_mood_based_recommendations, get_ingredient_based_recommendations
)

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
    .alert-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .success-card {
        background-color: #d1edff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=60)  # Cache expires after 60 seconds
def load_and_prepare_data():
    """Load and prepare data with caching"""
    # Check if data files exist, if not generate them
    if not os.path.exists('data/sample_orders.csv'):
        st.info("Generating sample data for the first time...")
        generator = DataGenerator()
        generator.save_sample_data()
    
    # Load data
    orders_df, menu_df, inventory_df, preferences_df = load_data()
    
    if orders_df is None:
        st.error("Error loading data. Please check if data files exist.")
        return None, None, None, None
    
    return orders_df, menu_df, inventory_df, preferences_df

def main():
    # Header
    st.markdown('<h1 class="main-header">🍽️ Sistem Manajemen Restoran AI</h1>', unsafe_allow_html=True)
    
    # Load data
    orders_df, menu_df, inventory_df, preferences_df = load_and_prepare_data()
    
    if orders_df is None:
        st.error("Failed to load data. Please check the data files.")
        return
    
    # Sidebar navigation
    st.sidebar.title("📊 Menu Navigasi")
    page = st.sidebar.selectbox(
        "Pilih Halaman:",
        ["🏠 Dashboard", "🍽️ Rekomendasi Menu", "📦 Pengelolaan Inventaris", "⚙️ Pengaturan"]
    )
    
    # Display selected page
    if page == "🏠 Dashboard":
        show_dashboard(orders_df, menu_df, inventory_df)
    elif page == "🍽️ Rekomendasi Menu":
        show_menu_recommendations(menu_df, orders_df, preferences_df)
    elif page == "📦 Pengelolaan Inventaris":
        show_inventory_management(inventory_df, orders_df, menu_df)
    elif page == "⚙️ Pengaturan":
        show_settings()

def show_dashboard(orders_df, menu_df, inventory_df):
    """Show main dashboard"""
    st.header("📊 Dashboard Utama")
    
    # Calculate metrics
    metrics = calculate_metrics(orders_df, inventory_df)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Pendapatan",
            value=format_currency(metrics['total_revenue']),
            delta=format_currency(metrics['weekly_revenue'])
        )
    
    with col2:
        st.metric(
            label="Total Pesanan",
            value=f"{metrics['total_orders']:,}",
            delta=f"{metrics['total_orders']//30:,} per bulan"
        )
    
    with col3:
        st.metric(
            label="Rata-rata Nilai Pesanan",
            value=format_currency(metrics['avg_order_value']),
            delta=""
        )
    
    with col4:
        st.metric(
            label="Stok Rendah",
            value=f"{metrics['low_stock_count']}",
            delta=f"{metrics['low_stock_percentage']:.1f}% dari total"
        )
    
    # Charts
    st.subheader("📈 Analisis Data")
    
    try:
        charts = create_simple_charts(orders_df, menu_df, inventory_df)
        
        # Display charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.pyplot(charts['trends'])
        
        with col2:
            st.pyplot(charts['popularity'])
        
        st.pyplot(charts['inventory'])
        
    except Exception as e:
        st.error(f"Error creating charts: {e}")
    
    # Inventory alerts
    st.subheader("🚨 Alert Inventaris")
    low_stock_items = get_low_stock_alerts(inventory_df)
    
    if len(low_stock_items) > 0:
        st.warning(f"⚠️ {len(low_stock_items)} item dengan stok rendah:")
        for _, item in low_stock_items.iterrows():
            st.write(f"• {item['ingredient_name']}: {item['current_stock']} {item['unit']}")
    else:
        st.success("✅ Semua stok dalam kondisi normal")
    
    # Recent orders
    st.subheader("📋 Pesanan Terbaru")
    recent_orders = orders_df.groupby('order_id').agg({
        'menu_name': lambda x: ', '.join(x),
        'total_price': 'sum',
        'order_date': 'first'
    }).sort_values('order_date', ascending=False).head(10)
    
    st.dataframe(recent_orders, use_container_width=True)

def create_simple_charts(orders_df, menu_df, inventory_df):
    """Create simple charts using matplotlib"""
    charts = {}
    
    # Daily orders trend
    daily_orders = orders_df.groupby('order_date').agg({
        'quantity': 'sum',
        'total_price': 'sum'
    }).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Quantity trend
    ax1.plot(daily_orders['order_date'], daily_orders['quantity'])
    ax1.set_title('Daily Order Quantity Trend')
    ax1.set_ylabel('Quantity')
    ax1.tick_params(axis='x', rotation=45)
    
    # Revenue trend
    ax2.plot(daily_orders['order_date'], daily_orders['total_price'])
    ax2.set_title('Daily Revenue Trend')
    ax2.set_ylabel('Revenue (IDR)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    charts['trends'] = fig
    
    # Menu popularity
    menu_popularity = orders_df.groupby('menu_name')['quantity'].sum().sort_values(ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    menu_popularity.plot(kind='bar', ax=ax)
    ax.set_title('Top 10 Most Popular Menu Items')
    ax.set_ylabel('Total Quantity Sold')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    charts['popularity'] = fig
    
    # Inventory status
    status_counts = inventory_df['status'].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%')
    ax.set_title('Inventory Status Distribution')
    charts['inventory'] = fig
    
    return charts

def show_menu_recommendations(menu_df, orders_df, preferences_df):
    """Show menu recommendations page"""
    st.header("🍽️ Rekomendasi Menu")
    
    # Recommendation type selection
    st.subheader("🎯 Jenis Rekomendasi")
    
    rec_type = st.selectbox(
        "Pilih jenis rekomendasi:",
        ["Berdasarkan Suasana Hati", "Berdasarkan Bahan", "Menu Terpopuler"]
    )
    
    if rec_type == "Berdasarkan Suasana Hati":
        mood = st.selectbox(
            "Pilih suasana hati:",
            ["comfort", "healthy", "quick", "refreshing", "sweet", "protein", "crispy"]
        )
        
        if st.button("🔍 Dapatkan Rekomendasi"):
            try:
                recommendations = get_mood_based_recommendations(menu_df, mood)
                
                if len(recommendations) > 0:
                    display_simple_recommendations(recommendations, f"Berdasarkan suasana hati: {mood}")
                else:
                    st.info("Tidak ada rekomendasi yang ditemukan")
            
            except Exception as e:
                st.error(f"Error dalam rekomendasi: {e}")
    
    elif rec_type == "Berdasarkan Bahan":
        ingredient = st.text_input("Masukkan nama bahan:", "")
        
        if st.button("🔍 Dapatkan Rekomendasi"):
            if ingredient:
                try:
                    recommendations = get_ingredient_based_recommendations(menu_df, ingredient)
                    
                    if len(recommendations) > 0:
                        display_simple_recommendations(recommendations, f"Berdasarkan bahan: {ingredient}")
                    else:
                        st.info(f"Tidak ada menu yang mengandung bahan '{ingredient}'")
                
                except Exception as e:
                    st.error(f"Error dalam rekomendasi: {e}")
            else:
                st.warning("Masukkan nama bahan terlebih dahulu")
    
    elif rec_type == "Menu Terpopuler":
        st.subheader("🍽️ Menu Terpopuler")
        
        # Get popular menus
        popular_menus = orders_df.groupby('menu_name').agg({
            'quantity': 'sum',
            'total_price': 'sum'
        }).sort_values('quantity', ascending=False).head(10)
        
        st.dataframe(popular_menus, use_container_width=True)

def display_simple_recommendations(recommendations_df, title):
    """Display recommendations in a simple format"""
    st.subheader(f"🍽️ {title}")
    
    for idx, rec in recommendations_df.iterrows():
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**{rec['name']}**")
                st.markdown(f"*{rec['category']}*")
                
                # Display ingredients and mood tags
                if 'ingredients' in rec:
                    ingredients = eval(rec['ingredients']) if isinstance(rec['ingredients'], str) else rec['ingredients']
                    st.markdown(f"**Bahan:** {', '.join(ingredients)}")
                
                if 'mood_tags' in rec:
                    mood_tags = eval(rec['mood_tags']) if isinstance(rec['mood_tags'], str) else rec['mood_tags']
                    st.markdown(f"**Mood:** {', '.join(mood_tags)}")
            
            with col2:
                st.markdown(f"**Harga:** {format_currency(rec['price'])}")
            
            st.divider()

def show_inventory_management(inventory_df, orders_df, menu_df):
    """Show inventory management page"""
    st.header("📦 Pengelolaan Inventaris")
    
    # Inventory overview
    st.subheader("📊 Overview Inventaris")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_items = len(inventory_df)
        st.metric("Total Item", total_items)
    
    with col2:
        low_stock_count = len(inventory_df[inventory_df['status'] == 'Low Stock'])
        st.metric("Stok Rendah", low_stock_count)
    
    with col3:
        total_value = inventory_df['current_stock'].sum()
        st.metric("Total Nilai Stok", format_currency(total_value))
    
    # Current inventory status
    st.subheader("📋 Status Inventaris Saat Ini")
    st.dataframe(inventory_df, use_container_width=True)
    
    # Low stock alerts
    st.subheader("🚨 Alert Stok Rendah")
    low_stock_items = get_low_stock_alerts(inventory_df)
    
    if len(low_stock_items) > 0:
        st.warning(f"⚠️ {len(low_stock_items)} item dengan stok rendah:")
        
        for _, item in low_stock_items.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**{item['ingredient_name']}**")
                    st.markdown(f"Stok saat ini: {item['current_stock']} {item['unit']}")
                
                with col2:
                    st.markdown(f"**Reorder point:** {item['reorder_point']}")
                
                with col3:
                    recommended_order = max(0, item['reorder_point'] - item['current_stock'])
                    st.markdown(f"**Rekomendasi order:** {recommended_order} {item['unit']}")
                
                st.divider()
    else:
        st.success("✅ Tidak ada alert stok rendah")
    
    # Ingredient usage analysis
    st.subheader("🔍 Analisis Penggunaan Bahan")
    
    # Simple ingredient usage based on menu sales
    ingredient_usage = {}
    
    for _, menu_item in menu_df.iterrows():
        menu_name = menu_item['name']
        ingredients = eval(menu_item['ingredients']) if isinstance(menu_item['ingredients'], str) else menu_item['ingredients']
        
        # Get total quantity sold for this menu
        menu_orders = orders_df[orders_df['menu_name'] == menu_name]
        total_quantity = menu_orders['quantity'].sum()
        
        for ingredient in ingredients:
            ingredient = ingredient.strip("'")
            if ingredient not in ingredient_usage:
                ingredient_usage[ingredient] = 0
            ingredient_usage[ingredient] += total_quantity
    
    # Display ingredient usage
    usage_df = pd.DataFrame([
        {'ingredient': k, 'usage': v} 
        for k, v in ingredient_usage.items()
    ]).sort_values('usage', ascending=False)
    
    st.dataframe(usage_df, use_container_width=True)

def show_settings():
    """Show settings page"""
    st.header("⚙️ Pengaturan")
    
    st.subheader("🔄 Regenerate Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Generate Sample Data Baru"):
            try:
                generator = DataGenerator()
                generator.save_sample_data()
                st.success("✅ Data sampel berhasil di-generate!")
                
                # Clear cache and reload
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Error dalam generate data: {e}")
    
    with col2:
        if st.button("🔄 Reload Data (Clear Cache)"):
            try:
                # Clear cache and reload
                st.cache_data.clear()
                st.success("✅ Cache berhasil di-clear! Data akan di-reload.")
                st.rerun()
            except Exception as e:
                st.error(f"Error dalam clear cache: {e}")
    
    st.subheader("📊 Informasi Sistem")
    
    # System information
    st.info("""
    **Sistem Manajemen Restoran AI**
    
    - **Versi:** 1.0.0 (Simple Version)
    - **Framework:** Streamlit + Python
    - **Visualization:** Matplotlib + Seaborn
    
    **Fitur Utama:**
    - Dashboard dengan analisis data real-time
    - Rekomendasi menu berdasarkan suasana hati dan bahan
    - Pengelolaan inventaris dengan alert stok rendah
    - Analisis penggunaan bahan berdasarkan penjualan menu
    
    **Data yang Tersedia:**
    - 7,000+ pesanan sampel
    - 10 menu items
    - 15 inventory items
    - 100 customer preferences
    """)

if __name__ == "__main__":
    main() 