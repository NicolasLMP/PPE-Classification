import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import os

def load_and_process_data():
    """Load and process fish detection data"""
    species_list = [
        "Anthias", "Atherine", "Bar européen", "Bogue", "Carangue", "Daurade Royale",
        "Daurade rose", "Eperlan", "Girelle", "Gobie", "Grande raie pastenague", "Grande vive",
        "Grondin", "Maquereau", "Merou", "Mostelle", "Mulet cabot", "Muraine", "Orphie",
        "Poisson scorpion", "Rouget", "Sole commune"
    ]
    
    csv_file = "fish_counts.csv"
    
    if not os.path.exists(csv_file):
        st.error(f"CSV file '{csv_file}' not found. Please run some detections first.")
        return None, None
    
    try:
        df = pd.read_csv(csv_file, encoding='latin1')
        df = df[df['ID'] != '---']
        
        # Calculate fish count for each species
        counts = []
        for specie in species_list:
            counts.append(len(df[df['Espèce'] == specie]))
        
        # Create DataFrame for sorting
        plot_df = pd.DataFrame({'Espèce': species_list, 'Nombre': counts})
        plot_df = plot_df.sort_values('Nombre', ascending=False)
        
        return df, plot_df
    
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None, None

def create_species_distribution_plot(plot_df):
    """Create species distribution bar chart"""
    fig, ax = plt.subplots(figsize=(16, 6))
    bars = ax.bar(plot_df['Espèce'], plot_df['Nombre'], color='skyblue')
    
    ax.set_ylabel('Nombre de détections')
    ax.set_xlabel('Espèce')
    ax.set_title('Distribution des espèces détectées')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig

def main():
    st.set_page_config(
        page_title="Fish Detection Metrics",
        layout="wide"
    )
    
    st.title("Fish Detection Metrics Report")
    
    # Load and process data
    df, plot_df = load_and_process_data()
    
    if df is None or plot_df is None:
        st.stop()
    
    # Summary statistics
    st.header("Summary Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Detections", len(df))
    
    with col2:
        species_with_detections = len(plot_df[plot_df['Nombre'] > 0])
        st.metric("Species Detected", species_with_detections)
    
    with col3:
        if len(df) > 0:
            most_common_species = plot_df.iloc[0]['Espèce']
            st.metric("Most Common Species", most_common_species)
    
    # Species distribution chart
    st.header("Species Distribution")
    
    if len(df) > 0:
        fig = create_species_distribution_plot(plot_df)
        st.pyplot(fig)
    else:
        st.info("No detections to display.")
    
    # Detailed table
    st.header("Detailed Counts")
    
    # Filter out species with zero detections for cleaner display
    filtered_df = plot_df[plot_df['Nombre'] > 0]
    
    if len(filtered_df) > 0:
        st.dataframe(filtered_df)
    else:
        st.info("No species detected yet.")
    
    # Raw data section
    with st.expander("View Raw Detection Data"):
        if len(df) > 0:
            st.dataframe(df)
            
            # Download button for raw data
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Raw Data as CSV",
                data=csv,
                file_name="fish_detections_export.csv",
                mime="text/csv"
            )
        else:
            st.info("No raw data available.")
    
    # Refresh button
    if st.button("Refresh Data"):
        st.rerun()

if __name__ == "__main__":
    main()