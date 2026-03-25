import speech_recognition as sr
from transformers import pipeline

def transcribe_audio_to_text(max_duration=240):
    """
    Captures audio from the microphone for a limited duration and converts it to text.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print(f"Listening... Speak for up to {max_duration} seconds.")
        try:
            # Adjust for ambient noise and record audio
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, phrase_time_limit=max_duration)
            print("Processing audio...")
            
            # Convert speech to text
            text = recognizer.recognize_google(audio)
            print(f"Transcribed Text: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results, error: {e}")
        return None

def summarize_text(text, max_length=100, min_length=30):
    """
    Summarizes the given text using a pre-trained transformer model.
    """
    try:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")  # Optimized for speed
        # Limit the input to a manageable size
        text = text[:1024]  # Only process the first 1024 characters
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error in summarization: {e}")
        return None

def main():
    """
    Main function to handle the AI Notes Maker workflow.
    """
    print("Welcome to AI Notes Maker!")
    
    # Step 1: Convert speech to text
    transcribed_text = transcribe_audio_to_text()
    
    if transcribed_text:
        # Step 2: Summarize the text
        print("\nSummarizing your notes...")
        summary = summarize_text(transcribed_text)
        
        if summary:
            print("\n--- Summary ---")
            print(summary)
        else:
            print("Failed to summarize the text.")
    else:
        print("No text to summarize.")

if __name__ == "__main__":
    main()
