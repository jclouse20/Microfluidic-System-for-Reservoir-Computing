import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

selecting = False
ix, iy = -1, -1
selected_areas = []
mode = 'rectangle'
rgb_data = defaultdict(list)
selection_in_progress = False
current_selection = None
scale_factor = 0.5

def select_area(event, x, y, flags, param):
    global ix, iy, selecting, current_selection, selection_in_progress, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        selecting = True
        selection_in_progress = True
        ix, iy = x, y
        current_selection = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting:
            if mode == 'rectangle':
                current_selection = ('rectangle', ix, iy, x, y)
            elif mode == 'circle':
                radius = int(np.sqrt((x - ix) ** 2 + (y - iy) ** 2))
                current_selection = ('circle', ix, iy, radius)

    elif event == cv2.EVENT_LBUTTONUP:
        selecting = False
        selection_in_progress = False
        if mode == 'rectangle':
            selected_areas.append(('rectangle', ix, iy, x, y))
        elif mode == 'circle':
            radius = int(np.sqrt((x - ix) ** 2 + (y - iy) ** 2))
            selected_areas.append(('circle', ix, iy, radius))
        print(f"Selected area: {selected_areas[-1]}")
        current_selection = None

def track_rgb_change(video_source=0):
    global rgb_data, mode, current_selection, selection_in_progress
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from video source.")
        return

    original_frame = frame.copy()
    frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    print("Press 'r' to select rectangle, 'c' for circle, and 'n' to finish selecting areas.")
    
    cv2.namedWindow('Select Area')
    
    cv2.setMouseCallback('Select Area', select_area)
    
    while True:
        display_frame = frame.copy()

        # Draw all previously selected areas with IDs
        for idx, area in enumerate(selected_areas, start=1):
            if area[0] == 'rectangle':
                _, x1, y1, x2, y2 = area
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Position the ID near the top-left corner of the rectangle
                cv2.putText(display_frame, str(idx), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2)
            elif area[0] == 'circle':
                _, cx, cy, radius = area
                cv2.circle(display_frame, (cx, cy), radius, (0, 255, 0), 2)
                # Position the ID to the right of the circle
                cv2.putText(display_frame, str(idx), (cx + radius + 10, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw the current selection
        if selection_in_progress and current_selection is not None:
            if current_selection[0] == 'rectangle':
                _, x1, y1, x2, y2 = current_selection
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            elif current_selection[0] == 'circle':
                _, cx, cy, radius = current_selection
                cv2.circle(display_frame, (cx, cy), radius, (255, 0, 0), 2)

        cv2.imshow('Select Area', display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):
            print("Rectangle mode")
            mode = 'rectangle'
        elif key == ord('c'):
            print("Circle mode")
            mode = 'circle'
        elif key == ord('n'):
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow('Select Area')

    # Scale selected areas back to original size
    scaled_selected_areas = []
    for area in selected_areas:
        if area[0] == 'rectangle':
            _, x1, y1, x2, y2 = area
            x1 = int(x1 / scale_factor)
            y1 = int(y1 / scale_factor)
            x2 = int(x2 / scale_factor)
            y2 = int(y2 / scale_factor)
            scaled_selected_areas.append(('rectangle', x1, y1, x2, y2))
        elif area[0] == 'circle':
            _, cx, cy, radius = area
            cx = int(cx / scale_factor)
            cy = int(cy / scale_factor)
            radius = int(radius / scale_factor)
            scaled_selected_areas.append(('circle', cx, cy, radius))

    frame_count = 0

    # Reset the capture to start from the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        for idx, area in enumerate(scaled_selected_areas, start=1):
            if area[0] == 'rectangle':
                _, x1, y1, x2, y2 = area
                x_start, x_end = sorted([x1, x2])
                y_start, y_end = sorted([y1, y2])
                roi = frame[y_start:y_end, x_start:x_end]
            elif area[0] == 'circle':
                _, cx, cy, radius = area
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.circle(mask, (cx, cy), radius, 255, -1)
                roi = cv2.bitwise_and(frame, frame, mask=mask)
                # Crop the ROI to the bounding rectangle of the circle
                x_start, x_end = max(cx - radius, 0), min(cx + radius, frame.shape[1])
                y_start, y_end = max(cy - radius, 0), min(cy + radius, frame.shape[0])
                roi = roi[y_start:y_end, x_start:x_end]

            # Check if ROI is valid
            if roi.size == 0:
                continue

            avg_color_per_row = np.average(roi, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            rgb_data[area].append(avg_color)
            
            # Draw the shapes around the selected areas with IDs
            if area[0] == 'rectangle':
                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                # Position the ID near the top-left corner of the rectangle
                cv2.putText(frame, str(idx), (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2)
            elif area[0] == 'circle':
                cv2.circle(frame, (cx, cy), radius, (0, 255, 0), 2)
                # Position the ID to the right of the circle
                cv2.putText(frame, str(idx), (cx + radius + 10, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Video Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    save_to_excel(rgb_data)
    plot_rgb_changes(rgb_data)

def save_to_excel(rgb_data):
    data_dict = {}
    max_length = max(len(values) for values in rgb_data.values())
    for i, (area, rgb_values) in enumerate(rgb_data.items()):
        r_vals = [rgb[2] for rgb in rgb_values]
        g_vals = [rgb[1] for rgb in rgb_values]
        b_vals = [rgb[0] for rgb in rgb_values]
        r_vals += [np.nan] * (max_length - len(r_vals))
        g_vals += [np.nan] * (max_length - len(g_vals))
        b_vals += [np.nan] * (max_length - len(b_vals))
        data_dict[f'Area_{i+1}_R'] = r_vals
        data_dict[f'Area_{i+1}_G'] = g_vals
        data_dict[f'Area_{i+1}_B'] = b_vals

    df = pd.DataFrame(data_dict)
    df.to_excel('rgb_data.xlsx', index=False)
    print(f"RGB data saved to rgb_data.xlsx")

def plot_rgb_changes(rgb_data):
    # Determine inputs and outputs based on initial RGB values
    threshold = 100  # Customizable threshold to distinguish inputs and outputs based on initial RGB values
    input_areas = []
    output_areas = []

    for area, rgb_values in rgb_data.items():
        initial_rgb = np.mean(rgb_values[0])  # Average of the first frame RGB values
        if initial_rgb > threshold:
            input_areas.append(area)
        else:
            output_areas.append(area)

    plt.figure(figsize=(12, 8))

    # Plot Inputs - Top plot
    plt.subplot(2, 1, 1)
    plt.title('Inputs: RGB Value Changes Over Time')

    for i, area in enumerate(input_areas):
        r_vals = [rgb[2] for rgb in rgb_data[area]]
        g_vals = [rgb[1] for rgb in rgb_data[area]]
        b_vals = [rgb[0] for rgb in rgb_data[area]]
        frames = range(len(r_vals))

        plt.plot(frames, r_vals, label=f'Area_{i+1}_R', color='red')
        plt.plot(frames, g_vals, label=f'Area_{i+1}_G', color='green')
        plt.plot(frames, b_vals, label=f'Area_{i+1}_B', color='blue')

    plt.xlabel('Frame')
    plt.ylabel('Average RGB Value')
    plt.legend(loc='best')

    # Plot Outputs - Bottom plot
    plt.subplot(2, 1, 2)
    plt.title('Outputs: RGB Value Changes Over Time')
    line_styles = ['-', '--', '-.', ':']  # Define line styles to cycle through for outputs

    for i, area in enumerate(output_areas):
        r_vals = [rgb[2] for rgb in rgb_data[area]]
        g_vals = [rgb[1] for rgb in rgb_data[area]]
        b_vals = [rgb[0] for rgb in rgb_data[area]]
        frames = range(len(r_vals))

        # Cycle through line styles for each output
        line_style = line_styles[i % len(line_styles)]

        plt.plot(frames, r_vals, label=f'Area_{i+1}_R', color='red', linestyle=line_style)
        plt.plot(frames, g_vals, label=f'Area_{i+1}_G', color='green', linestyle=line_style)
        plt.plot(frames, b_vals, label=f'Area_{i+1}_B', color='blue', linestyle=line_style)

    plt.xlabel('Frame')
    plt.ylabel('Average RGB Value')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.savefig('rgb_changes_input_output.png')
    plt.show()


# Replace "vid.mov" with the path to your video file
track_rgb_change(video_source=r"C:\microfluids\testvid.mov")
