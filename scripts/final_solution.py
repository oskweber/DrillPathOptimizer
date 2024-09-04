import os
import sys
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
from all_functions import (
    get_distance_dict,
    nearest_neighbour_v2,
    twoOpt,
    optimize_tsp_with_initial_solution,
    calculate_total_distance,
    solve_tsp_3opt,
    one_tree_lower_bound
)
import concurrent.futures
from multiprocessing import Queue, Manager

# Global variables to manage stopwatch and selected file paths
stopwatch_start = None
time_limit_start = None
time_limit_duration = None
selected_file_paths = []
queue = None

# Class to redirect output text to the GUI


class RedirectText(object):
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        # Insert string into the text widget
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)  # Scroll to the end
        self.text_widget.update_idletasks()  # Update the GUI

    def flush(self):
        pass  # Flush method is required for file-like object

# Function to choose a directory and list files in it


def choose_directory():
    directory_path = filedialog.askdirectory()
    file_listbox.delete(0, tk.END)
    if directory_path:
        # List all files in the selected directory
        all_files_and_dirs = os.listdir(directory_path)
        file_paths = [
            os.path.join(directory_path, f)
            for f in all_files_and_dirs
            if os.path.isfile(os.path.join(directory_path, f))
        ]
        # Display file names in the listbox
        for file_path in file_paths:
            file_listbox.insert(tk.END, os.path.basename(file_path))
        global selected_file_paths
        selected_file_paths = file_paths
    else:
        messagebox.showinfo("Information", "No directory selected.")

# Function to start a stopwatch timer


def start_stopwatch():
    global stopwatch_start
    stopwatch_start = datetime.now()
    update_stopwatch()

# Function to update the stopwatch every second


def update_stopwatch():
    if stopwatch_start is not None:
        now = datetime.now()
        elapsed = now - stopwatch_start
        elapsed_time = str(elapsed).split(".")[0]  # Format as H:MM:SS
        stopwatch_label.config(text=elapsed_time)
        # Schedule the function to run again after 1 second
        root.after(1000, update_stopwatch)
    else:
        stopwatch_label.config(text="00:00:00")

# Function to start a time limit countdown


def start_time_limit():
    global time_limit_start, time_limit_duration
    time_limit_start = datetime.now()
    try:
        available_time = float(available_time_entry.get())
        time_limit_duration = available_time
        check_time_limit()
    except ValueError:
        messagebox.showerror(
            "Invalid Input", "Please enter a valid number for the available time.")

# Function to check if the time limit has been reached


def check_time_limit():
    if time_limit_start is not None and time_limit_duration is not None:
        elapsed = datetime.now() - time_limit_start
        if elapsed.total_seconds() >= time_limit_duration * 60:  # Convert minutes to seconds
            time_limit_reached()
        else:
            root.after(1000, check_time_limit)  # Check again in 1 second

# Function to handle actions when the time limit is reached


def time_limit_reached():
    update_stopwatch()  # Update the stopwatch display to show stopped time
    messagebox.showinfo("Time Limit Reached",
                        "The specified time limit has been reached.")
    # Optionally stop any ongoing processes related to optimization here

# Function to optimize a file using TSP algorithms


def optimize_file(distance_dict, points, queue, total_time):
    try:
        start_time = time.time()
        queue.put("NN started\n")
        initial_tour = nearest_neighbour_v2(
            distance_dict=distance_dict, points=points)
        queue.put("NN completed, 2opt started\n")
        if len(points) < 1500:
            # If fewer than 1500 points, use 3-opt
            print("3-opt")
            second_stage_tour = solve_tsp_3opt(
                initial_tour=initial_tour, distance_dict=distance_dict, total=total_time)
            print("3opt completed\n")
            queue.put(
                f"3opt completed in {round(abs((start_time - time.time())/60),2)}\n")
        else:
            # Otherwise, use 2-opt
            second_stage_tour = twoOpt(initial_tour, distance_dict, total_time)
            queue.put(
                f"2opt completed in {round(abs((start_time - time.time())/60),2)}\n")

        end_time = time.time()
        elapsed_time_seconds = end_time - start_time
        elapsed_time_minutes = elapsed_time_seconds / 60
        print(f"Time left: {total_time - elapsed_time_minutes} minutes.")

        # If there is enough time left, optimize further using Gurobi
        if elapsed_time_minutes < total_time - 0.5:
            optimal_tour = (
                optimize_tsp_with_initial_solution(
                    distance_dict,
                    points,
                    second_stage_tour,
                    total_time - elapsed_time_minutes,
                ),
                False,
            )
        else:
            optimal_tour = (second_stage_tour, True)

        # Compute a lower bound using 1-tree
        lower_bound = one_tree_lower_bound(distance_dict, 0)
        return (optimal_tour, lower_bound)
    except Exception as e:
        queue.put(f"Error in optimize_file: {e}\n")
        raise

# Function to handle TSP optimization process


def TSPoptimization():
    try:
        # Get the available time from user input
        available_time = float(available_time_entry.get())
        time_limit_duration = available_time
        start_stopwatch()
    except ValueError:
        messagebox.showerror(
            "Invalid Input", "Please enter a valid number for the available time.")
        return

    status_text.delete(1.0, tk.END)
    distance_dicts = {}
    points_sets = {}
    points_dicts = {}
    # Generate distance dictionaries for each selected file
    for file_path in selected_file_paths:
        try:
            distance_dict, points_dict = get_distance_dict(file_path)
            distance_dicts[file_path] = distance_dict
            points_dicts[file_path] = points_dict
            points = set()
            for i, j in distance_dict.keys():
                points.add(i)
                points.add(j)
            points_sets[file_path] = points

            status_message = f"{file_path} distance dictionary and points set have been created.\n"
            queue.put(status_message)
            root.update_idletasks()
        except Exception as e:
            error_message = f"Error in generating distance dictionary and points set: {e}\n"
            queue.put(error_message)
            status_message = f"Error in {file_path}: {e}\n"
            queue.put(status_message)
            root.update_idletasks()
    update_stopwatch()

    # Use multiprocessing to handle optimization for multiple files
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_file = {
            executor.submit(
                optimize_file,
                distance_dicts[file_path],
                points_sets[file_path],
                queue,
                time_limit_duration,
            ): file_path
            for file_path in selected_file_paths
        }
        results = {}
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results[file_path] = result
                status_message = f"{file_path} optimization completed.\n"
            except Exception as exc:
                status_message = f"{file_path} generated an exception: {exc}\n"
            queue.put(status_message)
            root.update_idletasks()

    # Display optimization results in the GUI
    result_rows = [
        [os.path.basename(file_path) for file_path in results.keys()],
        [
            str(
                round(calculate_total_distance(
                    distance_dict=distance_dicts[file_path],
                    route=results[file_path][0][0],
                    nn=results[file_path][0][1],
                ))
            )
            for file_path in results.keys()
        ],
        [str(results[file_path][1]) for file_path in results.keys()]
    ]
    update_stopwatch()

    # Write results to output files
    for key, value in results.items():
        with open(f"{key[:-4]}_best_path.txt", "w") as file:
            if value[0][1]:
                for i, point in enumerate(value[0][0], start=1):
                    file.write(
                        f"{i}: {points_dicts[key][point].x_coordinate} {points_dicts[key][point].y_coordinate} \n"
                    )
            else:
                tuples_dict = {t[0]: t for t in value[0][0]}
                # Initialize the sorted list starting with the tuple that starts with 0
                tuple_sorted = [tuples_dict[0]]
                # Build the sorted list
                i = 0
                current = tuples_dict[0][1]
                while current in tuples_dict:
                    tuple_sorted.append(tuples_dict[current])
                    current = tuples_dict[current][1]
                    if current == 0 and i != 0:
                        break
                    i += 1

                rute_nn = [tavel[0] for tavel in tuple_sorted]
                for i, point in enumerate(rute_nn, start=1):
                    file.write(
                        f"{i}: {points_dicts[key][point].x_coordinate} {points_dicts[key][point].y_coordinate} \n"
                    )

    # Clear existing results in the tree view
    for i in tree.get_children():
        tree.delete(i)

    # Insert new results into the tree view
    for i in range(len(result_rows[0])):
        tree.insert("", "end", values=(
            result_rows[0][i], result_rows[1][i], result_rows[2][i]))

# Function to poll messages from the queue and print them


def poll_queue():
    while not queue.empty():
        msg = queue.get_nowait()
        print(msg)
    root.after(100, poll_queue)


# Main GUI setup
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Drill Path Optimizer")
    root.configure(bg="#e0f7fa")

    # Title label
    title_label = tk.Label(
        root,
        text="Drill Path Optimizer",
        font=("Helvetica", 24, "bold"),
        bg="#00695c",
        fg="white",
    )
    title_label.grid(row=0, column=0, columnspan=2, pady=10, padx=10)

    # Frame for file selection and buttons
    frame = tk.Frame(root, bg="#e0f7fa")
    frame.grid(row=1, column=1, pady=10, padx=10)

    # Button to choose directory
    choose_button = tk.Button(
        frame,
        text="Choose Directory",
        command=choose_directory,
        font=("Helvetica", 12),
        bg="#004d40",
        fg="white",
        activebackground="#00796b",
        activeforeground="white",
    )
    choose_button.grid(row=0, column=0, padx=10, pady=5)

    # Listbox to display selected files
    file_listbox = tk.Listbox(
        frame, width=50, height=10, font=("Helvetica", 12), bg="#b2dfdb", fg="#004d40"
    )
    file_listbox.grid(row=1, column=0, padx=10, pady=5)

    # Button to start optimization
    optimize_button = tk.Button(
        frame,
        text="Optimize",
        command=TSPoptimization,
        font=("Helvetica", 12),
        bg="#004d40",
        fg="white",
        activebackground="#00796b",
        activeforeground="white",
    )
    optimize_button.grid(row=2, column=0, padx=10, pady=5)

    # Frame to display results
    result_frame = tk.Frame(root, bg="#e0f7fa")
    result_frame.grid(row=2, column=0, columnspan=2, pady=10, padx=10)

    # Treeview to display results
    columns = ["File", "Total Distance", "Lower Bound"]
    tree = ttk.Treeview(result_frame, columns=columns,
                        show="headings", height=4)
    tree.pack(pady=10)

    # Configure treeview columns
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)

    # Style for treeview
    style = ttk.Style()
    style.configure(
        "Treeview.Heading",
        font=("Helvetica", 12, "bold"),
        background="#004d40",
        foreground="white",
    )
    style.configure(
        "Treeview",
        font=("Helvetica", 12),
        background="#b2dfdb",
        foreground="#004d40",
        fieldbackground="#b2dfdb",
    )

    # Frame for status and output
    status_frame = tk.Frame(root, bg="#e0f7fa")
    status_frame.grid(row=0, column=2, rowspan=3, pady=10, padx=10)

    # Text widget to display status messages
    status_text = tk.Text(
        status_frame,
        width=50,
        height=20,
        font=("Helvetica", 12),
        bg="#e0f7fa",
        fg="#004d40",
        wrap=tk.WORD,
    )
    status_text.pack()

    # Frame for time input
    time_entry_frame = tk.Frame(root, bg="#e0f7fa")
    time_entry_frame.grid(row=3, column=0, pady=5, padx=10)

    # Label and entry for time input
    time_label = tk.Label(
        time_entry_frame,
        text="Available Time (minutes):",
        bg="#e0f7fa",
        font=("Helvetica", 12),
    )
    time_label.pack(side=tk.LEFT)

    available_time_entry = tk.Entry(
        time_entry_frame, width=10, font=("Helvetica", 12))
    available_time_entry.pack(side=tk.LEFT)

    # Label to display stopwatch
    stopwatch_label = tk.Label(
        root, text="00:00:00", font=("Helvetica", 16), bg="#e0f7fa"
    )
    stopwatch_label.grid(row=4, column=0, pady=5, padx=10)

    # Redirect stdout to the GUI
    redirect_text = RedirectText(status_text)
    sys.stdout = redirect_text

    # Set up a queue for inter-process communication
    manager = Manager()
    queue = manager.Queue()

    # Start polling the queue for messages
    root.after(100, poll_queue)

    # Start the main GUI loop
    root.mainloop()
