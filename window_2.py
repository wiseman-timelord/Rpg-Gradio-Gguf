# Script: .\window_2.py

# Imports
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import yaml, time, os, sys, argparse, subprocess
from scripts.utility import fancy_delay

# Define the ramfs directory
RAMFS_DIR = '/mnt/ramfs'

class Watcher:
    DIRECTORY_TO_WATCH = RAMFS_DIR
    def __init__(self):
        self.observer = Observer()
    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(1)
        except:
            self.observer.stop()
            print("Observer stopped")
        self.observer.join()

class Handler(FileSystemEventHandler):
    def __init__(self):
        self.last_session_history = None
        self.last_sound_event = None

    def process(self, event):
        should_update_display = False  
        if event.src_path.endswith('persistent.yaml'):
            data = read_yaml()
            current_sound_event = data.get('sound_event', '')
            if current_sound_event != self.last_sound_event:
                self.last_sound_event = current_sound_event
                sound_file = f"./data/sounds/{current_sound_event}.wav"
                if os.path.exists(sound_file) and args.sound:
                    play_wav(sound_file)
            current_session_history = data.get('session_history', '')
            if current_session_history != self.last_session_history:
                self.last_session_history = current_session_history
                should_update_display = True  
            if should_update_display:  
                print(" ...changes detected, re-printing Display...\n")
                if args.sound:
                    play_wav(f"./data/sounds/change_detect.wav")  
                time.sleep(1)
                fancy_delay(3)
                display_interface()
                if args.tts:
                    speak_text(current_session_history)

    def on_modified(self, event):
        self.process(event)

def read_yaml(file_path=os.path.join(RAMFS_DIR, 'persistent.yaml')):
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: persistent.yaml not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading persistent.yaml: {e}")
        return None

def speak_text(text):
    subprocess.run(['espeak', text])

# Play the sample
def play_wav(filename):
    if args.sound:
        subprocess.run(['aplay', filename])

# main display
def display_interface():
    os.system('clear')
    data = read_yaml()
    if data is None:
        return
    human_name = data.get('human_name')
    agent_name = data.get('agent_name')
    agent_output_1 = data.get('agent_output_1')
    agent_emotion = data.get('agent_emotion')
    session_history = data.get('session_history')
    human_input = data.get('human_input')
    print("=" * 90)
    print("    ROLEPLAY SUMMARY")
    print("=" * 90, "=-" * 44)
    print(f" {human_name}'s Input:")
    print("-" * 90)
    print(f"\n {human_input}\n")
    print("=-" * 45, "=-" * 44)
    print(f" {agent_name}'s Response:")
    print("-" * 90)
    print(f"\n {agent_output_1}\n")
    print("=-" * 45, "=-" * 44)    
    print(f" {agent_name}'s State:")
    print("-" * 90)
    print(f"\n {agent_emotion}\n")  
    print("=-" * 45, "=-" * 44)   
    print(" Event History:")
    print("-" * 90)
    print(f"\n {session_history}\n")
    print("=-" * 45)
    print("\n Listening for changes to config.yaml...")

# End bit
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chat-VulkanLlama")
    parser.add_argument("--tts", action="store_true", help="Enable text-to-speech")
    parser.add_argument("--sound", action="store_true", help="Enable sound effects")
    args = parser.parse_args()
    display_interface()
    w = Watcher()
    w.run()
