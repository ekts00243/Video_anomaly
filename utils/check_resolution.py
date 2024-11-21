import cv2

def get_rtsp_resolution(rtsp_url):
    """
    Fetch and print the resolution of an RTSP stream.

    Parameters:
    - rtsp_url (str): The RTSP URL of the video stream.
    """
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Error: Cannot open RTSP stream: {rtsp_url}")
        return

    # Get frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Resolution of the RTSP stream: {frame_width}x{frame_height}")

    # Release the video capture object
    cap.release()

# Example usage
if __name__ == "__main__":
    rtsp_url = "rtsp://Onvif_test:Admin@123@10.7.100.1:554/live/D5A1BD2C-3462-4E10-8AEE-B7D24ABFA4CD"
    get_rtsp_resolution(rtsp_url)
