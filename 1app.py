import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
import json
import os


# Load model and scalers
model = joblib.load('urban_change_model.joblib')
coord_scaler = joblib.load('coord_scaler.joblib')
dist_scaler = joblib.load('dist_scaler.joblib')

def main():
    st.set_page_config(page_title="Urban Change Predictor", layout="wide")

    # Custom CSS for background image
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: url("https://imgix.brilliant-africa.com/Nairobi-National-Park-1.jpg?auto=format,enhance,compress&fit=crop&crop=entropy,faces,focalpoint&w=1880&h=740&q=30") no-repeat center center fixed;
                background-size: cover;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("🌆 Urban Change Prediction Dashboard")
    
    # Input Selection
    input_method = st.radio("Select Input Method:", ["📍 Manual Entry", "📁 Upload CSV"], horizontal=True)
    
    if input_method == "📍 Manual Entry":
        col1, col2 = st.columns(2)
        
        with col1:
            x = st.number_input("Longitude (x)", value=36.922205, format="%.6f")
            y = st.number_input("Latitude (y)", value=-1.336726, format="%.6f")
        
        with col2:
            road = st.number_input("Distance to Road (m)", value=412.73)
            airport = st.number_input("Distance to Airport (m)", value=1987.40)
            urban = st.number_input("Distance to Urban (m)", value=10142.31)
        
        input_df = pd.DataFrame({
            'x': [x], 'y': [y],
            'Distance_to_Road(Metres)': [road],
            'Distance_to_Airport(Metres)': [airport],
            'Distance_to_Urban_Areas(Metres)': [urban]
        })
    else:
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_file:
            input_df = pd.read_csv(uploaded_file)
            st.dataframe(input_df.head(8))
    if st.button("🚀 Predict"):
        st.session_state.predict_clicked = True

    if st.session_state.get("predict_clicked", False) and 'input_df' in locals():
        with st.spinner("Analyzing... Please wait."):

    # if st.button("🚀 Predict") and 'input_df' in locals():
    #     with st.spinner("Analyzing... Please wait."):
            # Scale features
            coords = coord_scaler.transform(input_df[['x', 'y']])
            dists = dist_scaler.transform(input_df.filter(regex='Distance_to_'))
            X = np.hstack([coords, dists])
            
            # Make predictions
            preds = model.predict(X)
            probas = model.predict_proba(X)[:, 1]  # Probability of Urban Change
            
            # Format results
            results = input_df.copy()
            results['Prediction'] = np.where(preds == 1, 'Urban Change', 'No Change')
            results['Probability'] = [f"{p:.1%}" for p in probas]
            results['Confidence'] = np.select(
                [probas > 0.7, probas > 0.4],
                ['High', 'Medium'],
                default='Low'
            )
            
            #View the output on a map
            # Create map centered on Nairobi
            m = folium.Map(
            location=[-1.286389, 36.817223],  # Center on Nairobi
            zoom_start=12,
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri World Imagery'
            )
            
            # 🗺️ Add Nairobi Boundary (GeoJSON)
            boundary_path = "nairobi_boundary.geojson"
            if os.path.exists(boundary_path):
                with open(boundary_path, "r", encoding="utf-8") as f:
                    nairobi_geojson = json.load(f)

            folium.GeoJson(
                nairobi_geojson,
                name="Nairobi Boundary",
                style_function=lambda feature: {
                    'fillColor': '#00000000',
                    'color': 'yellow',
                    'weight': 2,
                     },
                ).add_to(m)
                
            # Plot prediction points
            for _, row in results.iterrows():
                color = 'red' if row['Prediction'] == 'Urban Change' else 'cyan'
                folium.CircleMarker(
                    location=[row['y'], row['x']],
                    radius=8,
                    color=color,
                    fill=True,
                    fill_opacity=1,
                    popup=folium.Popup(f"""
                    <b>Prediction:</b> {row['Prediction']}<br>
                    <b>Probability:</b> {row['Probability']}<br>
                    <b>Confidence:</b> {row['Confidence']}
                        """, max_width=250)
                    ).add_to(m)

            # Optional: Add Layer Control if you want to toggle between base maps
            folium.LayerControl().add_to(m)

            # Add custom legend
            legend_html = '''
            <div style="
                position: fixed; 
                top: 5px; right: 10px; 
                width: 180px; 
                background-color: rgba(255, 255, 255, 0.8); 
                border: 1px solid #ccc; 
                border-radius: 10px;
                box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.3);
                z-index:9999; 
                font-size:14px;
                padding: 12px;
                ">
                <b>Legend</b><br><br>
                <i style="background: red; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i>&nbsp; Urban Change<br><br>
                <i style="background: cyan; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i>&nbsp; No Change
                </div>
                '''
            m.get_root().html.add_child(folium.Element(legend_html))

            # Add map information box (satellite source, prediction year)
            info_html = '''
            <div style="
                position: fixed; 
                top: 140px; right: 10px; 
                width: 180px; 
                background-color: rgba(255, 255, 255, 0.8); 
                border: 1px solid #ccc; 
                border-radius: 10px;
                box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.3);
                z-index:9998; 
                font-size:13px;
                padding: 12px;
                ">
                <b>Map Info</b><br>
                Imagery: Esri World Imagery<br>
                Prediction Year: 2024
                </div>
                '''
            m.get_root().html.add_child(folium.Element(info_html))

            # Apply styling and display results table
            def highlight_urban_change(row):
                return ['background-color: #90EE90' if row['Prediction'] == 'Urban Change' else ''] * len(row)
        
            st.dataframe(
                results.style
                    .apply(highlight_urban_change, axis=1)
                    .format({
                        'x': '{:.6f}', 'y': '{:.6f}',
                        'Distance_to_Road(Metres)': '{:.2f}',
                        'Distance_to_Airport(Metres)': '{:.2f}',
                        'Distance_to_Urban_Areas(Metres)': '{:.2f}'
                    })
                )
            
            # Download button
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download Results", csv, "urban_change_predictions.csv", "text/csv")


        # Show map in Streamlit
        st.subheader("🛰️ Nairobi Satellite View + Predicted Urban Change")
        st_folium(m, width=1000, height=600)
                
            
if __name__ == "__main__":
    main()
