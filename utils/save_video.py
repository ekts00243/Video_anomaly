import cv2
import time
import os
import sys

def record_rtsp_stream(rtsp_url, output_path, duration_minutes, fps=30):
    """
    Records an RTSP stream to a specified file.
    
    Parameters:
    - rtsp_url (str): The RTSP URL of the stream.
    - output_path (str): The path to save the recorded video.
    - duration_minutes (int): Duration to record the stream in minutes (2 to 10 minutes).
    - fps (int): Frames per second for recording (default: 30).
    """
    # Validate duration
    if not (2 <= duration_minutes <= 10):
        raise ValueError("Duration must be between 2 and 10 minutes.")
    
    # Open the RTSP stream
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise ValueError(f"Cannot open RTSP stream: {rtsp_url}")

    # Get the video frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    print(f"Recording started: {output_path}")
    start_time = time.time()
    duration_seconds = duration_minutes * 60

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from stream. Stopping recording.")
            break

        # Write the frame to the video file
        out.write(frame)

        # Calculate time left and display it
        elapsed_time = time.time() - start_time
        time_left = int(duration_seconds - elapsed_time)
        if time_left < 0:
            break

        minutes_left, seconds_left = divmod(time_left, 60)
        sys.stdout.write(f"\rTime left: {minutes_left:02d}:{seconds_left:02d} (MM:SS)")
        sys.stdout.flush()

        # Small sleep to avoid unnecessary CPU usage
        time.sleep(0.1)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nRecording completed: {output_path}")


# Example usage
if __name__ == "__main__":
    rtsp_url = "rtsp://Onvif_test:Admin@123@10.7.100.1:554/live/D5A1BD2C-3462-4E10-8AEE-B7D24ABFA4CD"
    output_path = "Video_Anomaly_Det/data/input_videos/recorded_video.mp4"
    duration_minutes = 2  # Change this value to 2-10 minutes
    record_rtsp_stream(rtsp_url, output_path, duration_minutes)
