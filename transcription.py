import torch # Deep learning framework
from transformers import pipeline # ML models
import os # file handling
import pandas as pd # data manipulation
import warnings
from pydub import AudioSegment


whisper = pipeline("automatic-speech-recognition", model="openai/whisper-base",  # load whisper model
                   device=0 if torch.cuda.is_available() else -1) # Use GPU if available

def transcribe_video(file_path):
    # First, extract the audio from the video file and save it as a WAV file
    audio_path = os.path.splitext(file_path)[0] + ".wav"
    try:
        video_clip = AudioSegment.from_file(file_path, format="mp4")
        video_clip.export(audio_path, format="wav")
    except Exception as e:
        print(f"Error converting {file_path}: {e}")
        return None  # Skip this file if conversion fails

    # Then, pass the new WAV file to the whisper pipeline
    try:
        audio = whisper(audio_path, return_timestamps=True) # transcribe audio ( Timestamps needed cuz it breaks without them for >30 sec audio)
        return audio["text"]
    
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        return None
    finally:
        # Clean up the temporary WAV file to save space
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
            
def transcribe_every_video(dir_name='data'): # function to transcribe a list of video files
    video_files = os.listdir(dir_name) # list all video files in the specified directory
    video_files = [os.path.join(dir_name, f) for f in video_files if f.endswith('.mp4')] # filter for video files
    transcriptions = {}
    n=len(video_files)
    if n == 0:
        raise ValueError(f"No .mp4 files found in directory: {dir_name}")
    for i, video in enumerate(video_files):
        print(f"Transcribing video {i+1}/{n}: {video}")
        transcription = transcribe_video(video)
        transcriptions[video] = transcription # store transcription in dictionary
    return transcriptions # return dictionary of transcriptions

def create_labeled_dataset(
        transcriptions, 
        labels_file = os.path.join('data', 'labels.csv'), 
        output_file='labeled_text_data.csv'
            ): 
    
    # function to create labeled dataset
    labels_df = pd.read_csv(labels_file) # read labels from CSV file
    
    # Check for mismatched videos between labels and transcriptions, sets are used to ignore order and duplicates
    unique_videos_in_transcriptions = set(labels_df['video'].apply(lambda x: os.path.basename(x)).unique().tolist())
    unique_videos_in_labels = set([os.path.basename(video) for video in transcriptions.keys()])
    if unique_videos_in_transcriptions != unique_videos_in_labels:
        if unique_videos_in_transcriptions.issubset(unique_videos_in_labels):
            warnings.warn("Some videos in the labels file are missing in the transcriptions.")
        elif unique_videos_in_labels.issubset(unique_videos_in_transcriptions):
            warnings.warn("Some videos in the transcriptions are missing in the labels file.")
        else:
            raise ValueError("Mismatch between videos in labels file and transcriptions.")
        
    labels_dict = pd.Series(labels_df.label.values,index=labels_df.video).to_dict()
    transcriptions = {video: (transcription, labels_dict.get(os.path.basename(video), 'unknown')) # unknown if label not found
                      for video, transcription in transcriptions.items()} # add labels to transcriptions
    df = pd.DataFrame(list(transcriptions.items()), columns=['video', 'transcription'])
    df.to_csv(output_file, index=False) # write DataFrame to CSV file
    
if __name__ == "__main__":
    transcriptions = transcribe_every_video()
    create_labeled_dataset(transcriptions)