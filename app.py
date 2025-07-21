
import streamlit as st
import os
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import webcolors
from sklearn.neighbors import NearestNeighbors

st.set_page_config(layout="wide", page_title="AI Fashion Finder")

# --- PATHS ---
BASE_DATA_DIR = "/kaggle/input/clothestry/clothes_tryon_dataset"
STYLE_CSV_PATH = "/kaggle/input/cloth-style-profile/cloth_style_profile.csv" 
DESC_CSV_PATH = "/kaggle/input/cloth-description-profile/cloth_description_profile.csv"
FEATURES_PATH = "/kaggle/working/embeddings_full_dataset.pkl"
MAP_PATH = "/kaggle/working/file_split_map.pkl"

# --- Helper Functions ---
def get_closest_color(rgb_tuple):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - rgb_tuple[0]) ** 2; gd = (g_c - rgb_tuple[1]) ** 2; bd = (b_c - rgb_tuple[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def get_color_name_from_rgb_str(rgb_str):
    try: return get_closest_color(eval(rgb_str))
    except: return "Unknown"

# --- Caching Functions for Performance ---
@st.cache_data
def load_data_and_embeddings():
    """Loads all data and creates a master DataFrame with a combined search column."""
    style_df = pd.read_csv(STYLE_CSV_PATH)
    desc_df = pd.read_csv(DESC_CSV_PATH)
    df = pd.merge(style_df, desc_df, on='filename', how='inner')
    
    with open(FEATURES_PATH, 'rb') as f: embeddings = np.array(pickle.load(f))
    file_map_df = pd.read_pickle(MAP_PATH)
    
    file_map_df['embedding'] = list(embeddings)
    final_df = pd.merge(df, file_map_df, on='filename', how='inner')
    
    # Create dynamic paths and the crucial combined search column
    final_df['color_name'] = final_df['dominant_color_rgb'].apply(get_color_name_from_rgb_str)
    final_df['cloth_path'] = final_df.apply(lambda r: os.path.join(BASE_DATA_DIR, r['split'], 'cloth', r['filename']), axis=1)
    final_df['model_path'] = final_df.apply(lambda r: os.path.join(BASE_DATA_DIR, r['split'], 'image', r['filename']), axis=1)
    final_df['search_text'] = (final_df['style_category'] + ' ' + final_df['description']).str.lower()
    
    return final_df

@st.cache_resource
def get_recommendation_model(_df):
    """Fits the NearestNeighbors model for cosine similarity recommendations."""
    embeddings = np.array(_df['embedding'].tolist())
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='cosine')
    neighbors.fit(embeddings)
    return neighbors

# --- Load all data ---
df = load_data_and_embeddings()
neighbors_model = get_recommendation_model(df)

# --- State Management for view navigation ---
if 'view' not in st.session_state: st.session_state.view = 'catalog'
if 'selected_item' not in st.session_state: st.session_state.selected_item = None

# --- UI Functions ---
def show_detail_view():
    """Renders the detailed view for a selected item and its recommendations."""
    item = st.session_state.selected_item
    st.header(f"Details for {item['style_category']}")
    if st.button("⬅️ Back to Catalog"):
        st.session_state.view = 'catalog'; st.rerun()

    col1, col2 = st.columns(2)
    with col1: st.image(item['model_path'], caption="Model View", use_column_width=True)
    with col2: st.image(item['cloth_path'], caption="Garment View", use_column_width=True)

    st.subheader("Description")
    st.write(f"**Style Category:** {item['style_category']}")
    st.write(f"**Dominant Color:** {item['color_name']}")
    st.write(f"**AI Description:** *{item['description']}*")
    st.markdown("---")
    
    st.header("✨ Visually Similar Items")
    item_embedding = item['embedding'].reshape(1, -1)
    _, indices = neighbors_model.kneighbors(item_embedding)
    rec_df = df.iloc[indices[0][1:]]
    rec_cols = st.columns(5)
    for i, (_, row) in enumerate(rec_df.iterrows()):
        with rec_cols[i]: st.image(row['cloth_path'], caption=row['style_category'], use_column_width=True)

def show_catalog_view():
    """Renders the main catalog with search, filters, and pagination."""
    st.sidebar.header("🔍 Search & Filter")
    
    # NEW: Text Search Engine
    search_query = st.sidebar.text_input("Search by description (e.g., 'black adidas shirt')")

    # Dropdown Filters
    style_options = ['All'] + sorted(df['style_category'].unique().tolist())
    color_options = ['All'] + sorted(df['color_name'].unique().tolist())
    sel_style = st.sidebar.selectbox("Filter by Style Category:", style_options)
    sel_color = st.sidebar.selectbox("Filter by Dominant Color:", color_options)

    # Filtering Logic
    filtered_df = df.copy()
    if search_query:
        search_terms = search_query.lower().split()
        for term in search_terms:
            filtered_df = filtered_df[filtered_df['search_text'].str.contains(term, na=False)]
    if sel_style != 'All': filtered_df = filtered_df[filtered_df['style_category'] == sel_style]
    if sel_color != 'All': filtered_df = filtered_df[filtered_df['color_name'] == sel_color]

    st.header(f"Browse Our Collection ({len(filtered_df)} items found)")
    
    # Pagination
    items_per_page = 20; total_pages = max(1, (len(filtered_df)-1)//items_per_page+1)
    page_num = st.sidebar.number_input(f'Page (1-{total_pages})', 1, total_pages, 1)
    start, end = (page_num - 1) * items_per_page, page_num * items_per_page
    paginated_df = filtered_df.iloc[start:end]
    
    # Display Grid
    cols = st.columns(5)
    for i, (idx, row) in enumerate(paginated_df.iterrows()):
        with cols[i % 5]:
            st.image(row['cloth_path'], use_column_width=True)
            st.caption(row['style_category'])
            if st.button("View Details & Similar", key=f"view_{idx}"):
                st.session_state.selected_item = row
                st.session_state.view = 'detail'; st.rerun()

# --- Main App Logic ---
st.title("👗 AI Fashion Finder")
st.markdown("Use the sidebar to search by text or filter by category. Click on any item to see details and find visually similar products.")

if st.session_state.view == 'detail':
    show_detail_view()
else:
    show_catalog_view()
