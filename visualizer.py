# visualizer.py
# This module is responsible for creating data visualizations.

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

def create_pie_chart(parent_frame, data_filepath, title):
    """
    Creates and embeds a matplotlib pie chart into a tkinter frame from a CSV file.

    Args:
        parent_frame (tkinter.Frame): The frame where the chart will be placed.
        data_filepath (str): Path to the CSV file containing the data.
        title (str): The title for the chart.
        
    Returns:
        FigureCanvasTkAgg: The canvas widget containing the chart.
    """
    try:
        # Read data from the specified CSV file
        df = pd.read_csv(data_filepath)
        
        # Filter out topics with zero count to avoid cluttering the chart
        plot_data = df[df['count'] > 0]
        
        if plot_data.empty:
            print(f"No data to plot for {title}.")
            return None

        # Create a figure and an axes for the plot
        # Using a constrained layout helps fit titles and labels.
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100, constrained_layout=True)
        
        # Generate the pie chart
        ax.pie(
            plot_data['count'], 
            labels=plot_data['topic'], 
            autopct='%1.1f%%', 
            startangle=140,
            shadow=True
        )
        ax.set_title(title, fontsize=14)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Embed the plot into the tkinter frame
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
        
        return canvas

    except FileNotFoundError:
        print(f"Error: Data file not found at '{data_filepath}'")
        return None
    except Exception as e:
        print(f"An error occurred during chart creation: {e}")
        return None
