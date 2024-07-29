import pandas as pd
import matplotlib.pyplot as plt

# Function to read data from an ODS file and plot it
def plot_from_ods(ods_file_path, sheet_name=0):
    # Read the ODS file
    df = pd.read_excel(ods_file_path, sheet_name=sheet_name, engine='odf')
    
    # Plotting the data
    plt.figure(figsize=(16, 9))  # Larger figure size

    for column in df.columns:
        plt.plot(df.index, df[column], label=column)

    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('MAP score', fontsize=14)
    
    # Adjusting the legend
    plt.legend(fontsize='large', loc='lower right')

    plt.grid(True)

    # Save the plot to a file with high resolution
    filename = 'thesis/results.png'
    plt.savefig(filename, dpi=300)  # High resolution for better quality

    # Display the plot
    plt.show()


# Usage
ods_file_path = 'thesis/Untitled 1.ods'  # Replace with your actual file path
plot_from_ods(ods_file_path)