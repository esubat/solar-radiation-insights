import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Load the dataset from the specified file path.
    """
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"File '{file_path}' not found.")

def preprocess_data(df):
    """
    Preprocess the dataset (e.g., convert Timestamp to datetime and set it as index).
    """
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    return df

def plot_time_series(df, variables_to_plot):
    """
    Plot time series analysis for the specified variables.
    """
    num_plots = len(variables_to_plot)
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, 6))
    for i, var in enumerate(variables_to_plot):
        sns.lineplot(x=df.index, y=var, data=df, ax=axes[i])
        axes[i].set_title(f'{var} over Time')
        axes[i].set_xlabel('Timestamp')
        axes[i].set_ylabel('Value')
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df):
    """
    Plot correlation matrix for the dataframe.
    """
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title('Correlation Matrix')
    plt.show()

def plot_wind_analysis(df):
    """
    Plot wind analysis including wind speed, wind direction, and wind speed vs. wind direction.
    """
    # Explore wind speed (WS, WSgust, WSstdev)
    wind_speed_columns = ['WS', 'WSgust', 'WSstdev']
    plt.figure(figsize=(8, 5))
    for col in wind_speed_columns:
        sns.histplot(df[col], bins=20, kde=True, label=col)
    plt.title('Distribution of Wind Speed')
    plt.xlabel('Wind Speed in m/s')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # Wind direction analysis
    plt.figure(figsize=(8, 5))
    sns.lineplot(x='Timestamp', y='WD', data=df)
    plt.title('Wind Direction over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Wind Direction (°)')
    plt.show()

    # Wind speed vs. wind direction analysis
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x='WD', y='WS', data=df)
    plt.title('Wind Speed vs Wind Direction')
    plt.xlabel('Wind Direction (°)')
    plt.ylabel('Wind Speed in m/s')
    plt.show()

def plot_temperature_analysis(df):
    """
    Plot temperature analysis including ambient temperature and module temperatures.
    """
    # Downsampling the data
    resampling_frequency = 'D'
    downsampled_data = df.resample(resampling_frequency).mean()

    # Temperature analysis
    temperature_data = downsampled_data[['Tamb', 'TModA', 'TModB']]
    plt.figure(figsize=(8, 4))
    sns.lineplot(x='Timestamp', y='Tamb', data=temperature_data, label='Ambient Temperature')
    sns.lineplot(x='Timestamp', y='TModA', data=temperature_data, label='Module A Temperature')
    sns.lineplot(x='Timestamp', y='TModB', data=temperature_data, label='Module B Temperature')
    plt.title('Temperature Analysis')
    plt.xlabel('Timestamp')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.show()

def plot_histograms(df):
    """
    Plot histograms for solar radiation, wind speed, and temperatures.
    """
    # Histograms for solar radiation variables
    solar_radiation_variables = ['GHI', 'DNI', 'DHI']
    plt.figure(figsize=(8, 4))
    for var in solar_radiation_variables:
        sns.histplot(df[var], bins=20, kde=True, label=var)
    plt.title('Histograms of Solar Radiation Variables')
    plt.xlabel('Solar Radiation (W/m²)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # Histogram for wind speed
    plt.figure(figsize=(8, 6))
    sns.histplot(df['WS'], bins=20, kde=True)
    plt.title('Histogram of Wind Speed')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Frequency')
    plt.show()

    # Histograms for temperatures
    plt.figure(figsize=(8, 4))
    for var in ['Tamb', 'TModA', 'TModB']:
        sns.histplot(df[var], bins=20, kde=True, label=var)
    plt.title('Histograms of Temperature Variables')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def plot_box_plots(df):
    """
    Plot box plots for solar radiation and temperature variables.
    """
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    sns.boxplot(data=df[['GHI', 'DNI', 'DHI']])
    plt.title('Box Plot of Solar Radiation Variables')
    plt.ylabel('Solar Radiation in W/m²')
    plt.subplot(2, 1, 2)
    sns.boxplot(data=df[['Tamb', 'TModA', 'TModB']])
    plt.title('Box Plot of Temperature Variables')
    plt.ylabel('Temperature (°C)')
    plt.tight_layout()
    plt.show()

def plot_scatter_plots(df):
    """
    Plot scatter plots for GHI vs. Tamb and WS vs. WSgust.
    """
    # Scatter plot: GHI vs. Tamb
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x='GHI', y='Tamb', data=df)
    plt.title('Scatter Plot: GHI vs. Tamb')
    plt.xlabel('GHI in W/m²')
    plt.ylabel('Tamb (°C)')
    plt.show()

    # Scatter plot: WS vs. WSgust
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x='WS', y='WSgust', data=df)
    plt.title('Scatter Plot: WS vs. WSgust')
    plt.xlabel('Wind Speed in m/s')
    plt.ylabel('Max Wind Gust Speed in m/s')
    plt.show()