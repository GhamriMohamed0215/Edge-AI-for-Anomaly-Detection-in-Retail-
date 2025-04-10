import tensorflow as tf
import cv2
import glob

# Define a function to apply color jittering to a video frame
def color_jitter(frame):
    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define random values for the color channel adjustments
    h_shift = tf.random.uniform([], -20, 20, dtype=tf.int32)
    s_shift = tf.random.uniform([], -50, 50, dtype=tf.int32)
    v_shift = tf.random.uniform([], -50, 50, dtype=tf.int32)

    # Apply the color channel adjustments to the HSV channels
    hsv[:, :, 0] += h_shift
    hsv[:, :, 1] += s_shift
    hsv[:, :, 2] += v_shift

    # Clip the HSV values to the valid range
    hsv = tf.clip_by_value(hsv, 0, 255)

    # Convert the HSV back to BGR color space
    bgr = cv2.cvtColor(hsv.numpy().astype('uint8'), cv2.COLOR_HSV2BGR)

    return bgr

# Define the path to the input videos
input_path = 'E:\\Anomaly-Detection-Dataset\\train\\Shoplifting1\\*.mp4'

# Loop through each video file in the input path
for video_file in glob.glob(input_path):
    # Open the video file
    cap = cv2.VideoCapture(video_file)

    # Get the video frame dimensions and frame rate
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the video writer to save the augmented video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_file = video_file.replace('.mp4', '_augmented.mp4')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Read each frame from the video, apply color jittering, and save the new frame to the output video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply color jittering to the frame
        augmented_frame = color_jitter(frame)

        # Write the augmented frame to the output video
        out.write(augmented_frame)

        # Display the augmented frame (optional)
        cv2.imshow('Augmented Frame', augmented_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and writer objects
    cap.release()
    out.release()

    # Close all windows
    cv2.destroyAllWindows()
