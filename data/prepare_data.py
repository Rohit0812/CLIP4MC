import os
import pickle
import yt_dlp
import cv2
import ffmpeg
import numpy as np
from transformers import AutoTokenizer

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def download_video(video_url: str, output_file: str):
    """Downloads the best video and audio and merges them into a single file."""
    ydl_opts = {
        "format": "bestvideo",
        "outtmpl": output_file,
        "merge_output_format": "mp4",  # Ensure merged MP4 output
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

def clip_video(input_file: str, start_time: int, end_time: int, output_file: str):
    """Clips the video using OpenCV."""
    ffmpeg.input(input_file, ss=start_time, to=end_time).output(output_file, vcodec="libx264", acodec="aac").run()
    """
    cap = cv2.VideoCapture(input_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    start_frame = start_time * fps
    end_frame = end_time * fps

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    """

def video_to_numpy(video_path: str):
    """Converts video frames to a numpy array."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Resize frame to (160, 256)
        frame = cv2.resize(frame, (256, 160), interpolation=cv2.INTER_AREA)
        # Convert frame to (C, H, W) -> (3, 160, 256)
        frame = np.transpose(frame, (2, 0, 1))
        frames.append(frame)

    cap.release()
    return np.array(frames)

def save_as_pickle(video_data, text_data, size, pickle_path, pickle_output_ti, pickle_output_size):
    """Saves video and text data as pickle files."""
    with open(pickle_path, "wb") as f:
        pickle.dump(video_data, f)

    with open(pickle_output_ti, "wb") as f:
        pickle.dump(text_data, f)

    with open(pickle_output_size, "wb") as f:
        pickle.dump(size, f)

def prepare_data(dataset, output_dir, offset_no=0, total_rows=20):
    """Prepares data for the dataset."""
    i = offset_no
    for item in dataset['train'].select(range(offset_no, offset_no + total_rows)):
        try:
            video_url = f"https://www.youtube.com/watch?v={item['vid']}"
            downloaded_video = "full_video.mp4"
            clipped_video = "clipped_video.mp4"
            pickle_output_vi = os.path.join(output_dir, f"data_dir_{i}/video_input.pkl")
            pickle_output_ti = os.path.join(output_dir, f"data_dir_{i}/text_input.pkl")
            pickle_output_size = os.path.join(output_dir, f"data_dir_{i}/size.json")

            # Tokenize text
            encoded_data = tokenizer(item['transcript clip'], truncation=True, return_tensors="pt", 
                                    max_length=tokenizer.model_max_length, padding="max_length")
            text = {"tokens": encoded_data["input_ids"].squeeze(0)}
            size = item['size']

            # Ensure output directory exists
            os.makedirs(os.path.dirname(pickle_output_vi), exist_ok=True)

            # Step 1: Download video
            download_video(video_url, downloaded_video)

            # Step 2: Clip video
            clip_video(downloaded_video, item['begin position'], item['end position'], clipped_video)

            # Step 3: Convert video to numpy and save as pickle
            video_data = video_to_numpy(clipped_video)
            save_as_pickle(video_data, text, size, pickle_output_vi, pickle_output_ti, pickle_output_size)

            # Step 4: Cleanup
            os.remove(downloaded_video)
            os.remove(clipped_video)
            
            i += 1
            print(f"Processed data_dir_{i}")
        except Exception as e:
            print(f"Error processing data_dir_{i}: {e}")

# Example usage
if __name__ == "__main__":
    from datasets import load_from_disk

    # Load dataset
    dataset = load_from_disk("../dataset/")

    # Prepare data
    prepare_data(dataset, "../dataset")