import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *

def visualize_data(df):
    """
    Visualize different analyses based on user selection.
    """
    st.title('Solar Radiation Data Analysis')

    analysis_option = st.sidebar.selectbox('Select Analysis Option', [
        'Time Series Analysis',
        'Correlation Analysis',
        'Wind Analysis',
        'Temperature Analysis',
        'Histograms',
        'Box Plots',
        'Scatter Plots',
    ])

    if analysis_option == 'Time Series Analysis':
        st.header('Time Series Analysis')
        variables_to_plot = st.multiselect('Select Variables to Plot', ['GHI', 'DNI', 'DHI', 'Tamb'])
        if variables_to_plot:
            plot_time_series(df, variables_to_plot)

    elif analysis_option == 'Correlation Analysis':
        st.header('Correlation Analysis')
        plot_correlation_matrix(df)

    elif analysis_option == 'Wind Analysis':
        st.header('Wind Analysis')
        plot_wind_analysis(df)

    elif analysis_option == 'Temperature Analysis':
        st.header('Temperature Analysis')
        plot_temperature_analysis(df)

    elif analysis_option == 'Histograms':
        st.header('Histograms')
        plot_histograms(df)

    elif analysis_option == 'Box Plots':
        st.header('Box Plots')
        plot_box_plots(df)

    elif analysis_option == 'Scatter Plots':
        st.header('Scatter Plots')
        plot_scatter_plots(df)

def main():
    # Load data
    df = load_data("./src/data/sierraleone-bumbuna.csv")

    # Preprocess data
    df = preprocess_data(df)

    # Visualize data based on user selection
    visualize_data(df)

if __name__ == '__main__':
    main()
