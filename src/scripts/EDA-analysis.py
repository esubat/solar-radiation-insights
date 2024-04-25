import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/benin-malanville.csv")

print(df.head())
df.info()

#statistics Summary
stat_summary = df.describe()
print(stat_summary)



# Data Quality Check and Data Cleaning

# looking missing values
missing_data = df.isnull().sum()
print(missing_data)

# negative values
negative_values = df[df.select_dtypes(include='number') < 0]
if not negative_values.empty:
    print("Negative values found:")
    print(negative_values)

#looking for outliers 
for col in df.select_dtypes(include='number').columns:
    std_dev = df[col].std()
    mean = df[col].mean()
    outliers = df[(df[col] > mean + 3 * std_dev) | (df[col] < mean - 3 * std_dev)]
    if not outliers.empty:
        print(f"Outliers found in {col}:")
        print(outliers)


# Time Series Analysis

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp' , inplace = True )

variables = ['GHI', 'DNI', 'DHI', 'Tamb']

num_plots = len(variables)
fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, 6))

#iterate over variables to be plotted
for i, var in enumerate(variables):
    sns.lineplot(x=df.index, y=var, data=df, ax=axes[i])
    axes[i].set_title(f'{var} over Time')
    axes[i].set_xlabel('Timestamp')
    axes[i].set_ylabel('Value')

plt.tight_layout()
plt.show()


#correlation Analysis

correlation_matrix = df.corr()

# Print correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Matrix')
plt.show()


# Wind Analysis
# Explore wind speed (WS, WSgust, WSstdev) 
# and wind direction (WD, WDstdev) data 

wind_speed_columns = ['WS', 'WSgust', 'WSstdev']
plt.figure(figsize=(8, 5))

for col in wind_speed_columns:
    sns.histplot(df[col], bins=20, kde=True, label=col)

    
plt.title('Distribution of Wind Speed')
plt.xlabel('Wind Speed in m/s')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Wind Direction Analysis
plt.figure(figsize=(8, 5))
sns.lineplot(x='Timestamp', y='WD', data=df)
plt.title('Wind Direction over Time')
plt.xlabel('Timestamp')
plt.ylabel('Wind Direction (°)')
plt.show()

# Wind Speed vs. Wind Direction Analysis
plt.figure(figsize=(6, 4))
sns.scatterplot(x='WD', y='WS', data=df)
plt.title('Wind Speed vs Wind Direction')
plt.xlabel('Wind Direction (°)')
plt.ylabel('Wind Speed in m/s')
plt.show()


# Downsampling the data

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True) 

resampling_frequency = 'D'

downsampled_data = df.resample(resampling_frequency).mean()  

print(downsampled_data)



# Temperature Analysis: 

# we will use the down sampled data to minimize the runtime
temperature_data = downsampled_data[['Tamb', 'TModA', 'TModB']]

# Temperature Analysis
plt.figure(figsize=(8, 4))
sns.lineplot(x='Timestamp', y='Tamb', data=temperature_data, label='Ambient Temperature')
sns.lineplot(x='Timestamp', y='TModA', data=temperature_data, label='Module A Temperature')
sns.lineplot(x='Timestamp', y='TModB', data=temperature_data, label='Module B Temperature')
plt.title('Temperature Analysis')
plt.xlabel('Timestamp')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()

# Temperature Comparison
plt.figure(figsize=(6, 4))
sns.scatterplot(x='Tamb', y='TModA', data=temperature_data, label='Module A')
sns.scatterplot(x='Tamb', y='TModB', data=temperature_data, label='Module B')
plt.title('Temperature Comparison')
plt.xlabel('Ambient Temperature (°C)')
plt.ylabel('Module Temperature (°C)')
plt.legend()
plt.show()

# Correlation Analysis
correlation_matrix = temperature_data[['Tamb', 'TModA', 'TModB']].corr()
print("Correlation Matrix:")
print(correlation_matrix)


# Histograms: 

# Histograms for Solar Radiation Variables (GHI, DNI, DHI)
solar_radiation_variables = ['GHI', 'DNI', 'DHI']

plt.figure(figsize=(8, 4))

for var in solar_radiation_variables:
    sns.histplot(data[var], bins=20, kde=True, label=var)
    
plt.title('Histograms of Solar Radiation Variables')
plt.xlabel('Solar Radiation (W/m²)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Histogram for Wind Speed (WS)
plt.figure(figsize=(8, 6))
sns.histplot(data['WS'], bins=20, kde=True)
plt.title('Histogram of Wind Speed')
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Frequency')
plt.show()

# Histograms for Temperatures (Tamb, TModA, TModB)
temperature_variables = ['Tamb', 'TModA', 'TModB']
plt.figure(figsize=(8, 4))
for var in temperature_variables:
    sns.histplot(data[var], bins=20, kde=True, label=var)
plt.title('Histograms of Temperature Variables')
plt.xlabel('Temperature (°C)')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# Box Plots: 

plt.figure(figsize=(8, 6))

# Solar Radiation Variables
plt.subplot(2, 1, 1)
sns.boxplot(data=df[['GHI', 'DNI', 'DHI']])
plt.title('Box Plot of Solar Radiation Variables')
plt.ylabel('Solar Radiation in W/m²')

# Temperature Variables
plt.subplot(2, 1, 2)
sns.boxplot(data=df[['Tamb', 'TModA', 'TModB']])
plt.title('Box Plot of Temperature Variables')
plt.ylabel('Temperature (°C)')

plt.tight_layout()
plt.show()



# Scatter Plots:

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
plt.xlabel('Wind Speed in m/s ')
plt.ylabel('Max Wind Gust Speed in m/s ')
plt.show()



# CleaningThe Data
df.dropna(inplace=True)
print("Cleaned Data Shape:", df.shape)

df.info()
df.isnull().sum()