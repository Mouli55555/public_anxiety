# main_app.py
# This is the main application file. It builds the GUI and orchestrates the program flow.

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import ttk, filedialog, messagebox, simpledialog

# Import our custom modules
import config
import data_processor
import visualizer
from predictor import fuzzy_predictor # <-- Import the new fuzzy predictor instance

class AnxietyAnalysisApp(tk.Tk):
    """
    The main class for the Anxiety Detection System application.
    """
    def __init__(self):
        super().__init__()
        
        self.title("Anxiety Detection System")
        self.geometry("800x600")
        
        # --- Internal State ---
        self.current_chart_canvas = None

        # --- UI Initialization ---
        self._create_widgets()

    def _create_widgets(self):
        # Use a main frame to better organize the layout
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # Split the window into two main parts: controls and display
        left_frame = ttk.Frame(main_frame, width=200)
        left_frame.pack(side='left', fill='y', padx=(0, 10))
        
        self.right_frame = ttk.Frame(main_frame)
        self.right_frame.pack(side='right', fill='both', expand=True)
        
        # --- Left Frame: Controls ---
        controls_labelframe = ttk.LabelFrame(left_frame, text="Controls", padding="10")
        controls_labelframe.pack(fill='x', pady=5)
        
        # Button to run the data processing
        process_button = ttk.Button(
            controls_labelframe, 
            text="1. Process Tweet Data", 
            command=self.run_data_processing
        )
        process_button.pack(fill='x', pady=5)
        
        # --- Updated section for Fuzzy Prediction ---
        prediction_labelframe = ttk.LabelFrame(left_frame, text="Fuzzy Anxiety Analysis", padding="10")
        prediction_labelframe.pack(fill='x', pady=10)

        predict_button = ttk.Button(
            prediction_labelframe,
            text="2. Analyze Tweet Anxiety",
            command=self.run_tweet_analysis
        )
        predict_button.pack(fill='x', pady=5)
        
        # Separator
        ttk.Separator(left_frame, orient='horizontal').pack(fill='x', pady=10)

        # Label and buttons for community analysis
        community_labelframe = ttk.LabelFrame(left_frame, text="View Community Analysis", padding="10")
        community_labelframe.pack(fill='x', pady=5)

        for i in range(config.NUM_COMMUNITIES):
            community_id = i + 1
            btn = ttk.Button(
                community_labelframe,
                text=f"Community {community_id}",
                # Use a lambda to pass the community_id to the command
                command=lambda cid=community_id: self.display_community_analysis(cid)
            )
            btn.pack(fill='x', pady=3)
            
        # --- Right Frame: Display Area ---
        self.display_title_label = ttk.Label(
            self.right_frame, 
            text="Welcome to the Anxiety Detection System!", 
            font=("Helvetica", 16)
        )
        self.display_title_label.pack(pady=10)
        
        self.chart_frame = ttk.Frame(self.right_frame)
        self.chart_frame.pack(fill='both', expand=True)
        
        # Initial message in the chart frame
        self.initial_message = ttk.Label(
            self.chart_frame,
            text="\nPlease run the data processing first,\nthen select a community to view its analysis.",
            justify="center",
            font=("Helvetica", 12)
        )
        self.initial_message.pack(pady=20)
        
    def run_data_processing(self):
        """
        Callback for the 'Process Tweet Data' button.
        """
        # Ask for confirmation as this can take a moment
        is_confirmed = messagebox.askyesno(
            "Confirm Processing",
            f"This will analyze the '{config.DATA_FILE_PATH}' file and generate new result files.\n\nDo you want to continue?"
        )
        
        if is_confirmed:
            success = data_processor.process_and_save_all_communities()
            if success:
                messagebox.showinfo("Success", "Data processing completed successfully!")
            else:
                messagebox.showerror("Error", f"Data processing failed. Please check the console and ensure '{config.DATA_FILE_PATH}' exists.")

    def run_tweet_analysis(self):
        """
        Callback for the 'Analyze Tweet Anxiety' button.
        Prompts user for a tweet and displays the fuzzy logic-based anxiety score.
        """
        # Ask the user for a full tweet or sentence
        tweet_text = simpledialog.askstring(
            "Tweet Input", 
            "Please enter a tweet or sentence to analyze:",
            parent=self
        )
        
        if not tweet_text:
            return # User cancelled
            
        # Run the prediction using the new fuzzy predictor
        predicted_score = fuzzy_predictor.predict(tweet_text)
        
        # Display the result in a more descriptive message box
        messagebox.showinfo(
            "Fuzzy Analysis Result",
            f"Input Text:\n'{tweet_text}'\n\n"
            f"The predicted anxiety score is: {predicted_score:.2f} / 100"
        )

    def display_community_analysis(self, community_id):
        """
        Displays the pie chart for the selected community.
        """
        data_filepath = os.path.join(config.OUTPUT_DIR, f'community_{community_id}_analysis.csv')

        if not os.path.exists(data_filepath):
            messagebox.showwarning(
                "File Not Found",
                f"Analysis file for Community {community_id} not found.\nPlease run the data processing first."
            )
            return

        # Clear any existing content in the right frame
        self.clear_display_frame()

        # Update the title
        self.display_title_label.config(text=f"Analysis for Community {community_id}")

        # Create and display the new chart
        self.current_chart_canvas = visualizer.create_pie_chart(
            self.chart_frame, 
            data_filepath, 
            f'Topic Distribution in Community {community_id}'
        )

    def clear_display_frame(self):
        """
        Clears the chart frame of any widgets.
        """
        # Destroy the initial message if it's there
        if self.initial_message:
            self.initial_message.destroy()
            self.initial_message = None

        # Destroy the previous chart canvas if it exists
        if self.current_chart_canvas:
            self.current_chart_canvas.get_tk_widget().destroy()
            self.current_chart_canvas = None

if __name__ == "__main__":
    app = AnxietyAnalysisApp()
    app.mainloop()


