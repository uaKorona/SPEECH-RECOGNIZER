import os
import math
from openai import OpenAI
from pydub import AudioSegment


# Define max file size (in bytes) for each chunk - 24 MB
MAX_CHUNK_SIZE = 24 * 1024 * 1024  # 24 MB

# Set your OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def split_audio(file_path, max_chunk_size=MAX_CHUNK_SIZE):
    """Split audio into chunks of max_chunk_size (in bytes)."""
    audio = AudioSegment.from_file(file_path)
    chunk_duration_ms = max_chunk_size / (audio.frame_rate * audio.frame_width * audio.channels) * 1000
    chunks = [audio[i:i + chunk_duration_ms] for i in range(0, len(audio), int(chunk_duration_ms))]
    return chunks

def format_time(milliseconds):
    """Convert milliseconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(milliseconds / (1000 * 60 * 60))
    minutes = int((milliseconds % (1000 * 60 * 60)) / (1000 * 60))
    seconds = int((milliseconds % (1000 * 60)) / 1000)
    milliseconds = int(milliseconds % 1000)

    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def parse_verbose_json_to_srt(json_response, start_time_offset=0):
    """Parse verbose_json response to SRT format with start_time_offset adjustment."""
    srt_output = []
    subtitle_index = 1

    for segment in json_response.segments:
        # Calculate start and end times with offset
        start_time = segment.start * 1000 + start_time_offset
        end_time = segment.end * 1000 + start_time_offset
        text = segment.text.strip()

        # Format SRT entry
        srt_entry = f"{subtitle_index}\n"
        srt_entry += f"{format_time(start_time)} --> {format_time(end_time)}\n"
        srt_entry += f"{text}\n"
        srt_output.append(srt_entry)
        
        subtitle_index += 1

    # print(f"SRT chunk {subtitle_index}", text)
    return "\n".join(srt_output)

def transcribe_audio_chunks(chunks, output_srt_path="transcription.srt"):
    """Transcribe each chunk and save the SRT results to a file."""
    transcription_result = []
    cumulative_start_time = 0  # Track the start time in milliseconds

    for i, chunk in enumerate(chunks):
        # Save each chunk as a temporary file
        temp_chunk_path = f"temp_chunk_{i}.mp3"
        chunk.export(temp_chunk_path, format="mp3")
        
        try:
            # Open the file and send it for transcription
            with open(temp_chunk_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(model="whisper-1", language="uk", file=audio_file, response_format="verbose_json")
                # Convert JSON response to SRT format with cumulative start time adjustment
                srt_text = parse_verbose_json_to_srt(response, cumulative_start_time)
                transcription_result.append(srt_text)

                # Update cumulative start time based on the chunk's duration
                cumulative_start_time += len(chunk)
                
        
        except Exception as e:
            print(f"An error occurred during transcription of chunk {i}:", e)
            break
        
        # Delete the temporary chunk file after processing
        finally:
            os.remove(temp_chunk_path)
    
    # Write combined SRT transcription to the file
    with open(output_srt_path, "w", encoding="utf-8") as srt_file:
        srt_file.write("\n".join(transcription_result))
    
    print(f"Transcription saved to {output_srt_path}")

def compress_and_convert_audio(input_path, wav_output_path, mp3_output_path, sample_rate=16000, channels=1, bitrate="32k"):
    """Compresses an audio file and exports it in both WAV and MP3 formats."""
    audio = AudioSegment.from_file(input_path)
    
    # Set to mono and adjust sample rate
    audio = audio.set_frame_rate(sample_rate).set_channels(channels)
    
    # Export compressed WAV file
    # audio.export(wav_output_path, format="wav", bitrate=bitrate)
    # print(f"Compressed WAV audio saved to {wav_output_path}")
    
    # Export compressed MP3 file
    audio.export(mp3_output_path, format="mp3", bitrate=bitrate)
    print(f"Compressed MP3 audio saved to {mp3_output_path}")

def speechRecognize():
    # Get the current directory of the script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    audio_file_path = os.path.join(current_dir, "audio.wav")

    if os.path.isfile(audio_file_path):
        compress_and_convert_audio("audio.wav", "audio_compressed.wav", "audio_compressed.mp3")
        audio_file_path = os.path.join(current_dir, "audio_compressed.mp3")
        
        file_size = os.path.getsize(audio_file_path)
        
        # Load the audio file as a single chunk if within 24MB or split into chunks if larger
        if file_size > MAX_CHUNK_SIZE:
            print("File is larger than 24MB. Splitting and transcribing in parts...")
            chunks = split_audio(audio_file_path)
        else:
            print("File is within the 24MB limit. Transcribing directly...")
            chunks = [AudioSegment.from_file(audio_file_path)]  # Treat entire file as a single chunk
        
        # Transcribe the chunks
        transcribe_audio_chunks(chunks)
    else:
        print("File 'audio.wav' does not exist in the directory.")
        

# Run the function when the file is executed
if __name__ == "__main__":
    speechRecognize()
