# Chat-Ubuntu-Gguf
- Status: Alpha

### Project Details
The Reinvention of my WSL based ChatBot....
- This one will start off simple but innovative, then expand from there. 
- Decision to integrate gradio from the start, thus freed up a script.
- Programmed towards Ubuntu 24 with Venv.
- Programmed towards CPU, for simpler memory management. there will be no use of, opencl or vulkan, or gpu for now.
- windows merged; by using gradio interface, removed window_2. and Window 1 became main_script. Window_2 will now be the "./scripts/interface.py" including, text and gradio interface, gradio interface displaying in the popup browser window.
- Programmed towards Gguf models only, maintained in "./models"; The model must now have a relating "./models/model_config.json" (useually supplied with the model). the program needs to scan this folder upon start, and utilize whatever model is there.
- A text window and a graphical window. gradio interface for the chat interface. Engine window as terminal with library output shown and any necessary debug info; gradio interface is in runspace, and main scripts return to bash upon exit, and exiting the menu on the bash will shutdown all required things. the bash identifies the process upon launching the gradio interface; so long as people follow logical exit, then it will correctly close the gradio process in runspace). 
- additional scripts are generated as required, and placed in data folder, such as the yaml for persistent settings.
- we have 5 python scripts in total between "./" and "./scripts" and "./data", plush the bash script, and the rest, generates or downloads.
- the terminal checking relevant folders for required files and performing any required maintenance, creating additional files as required, all before launching the gradio interface.

### Project Plans
3) Explore integrating stable diffusion models for AI-generated maps and scenarios, there are some in gguf. If maps were able to be generated, and the user types in the name of the place to go, then little else need be done other than 

### FEATURES
- Optimized code, 5 main scripts, creation of required folders.
- Running only with single GGUF model, featuring llama python.
- Terminal and Gradio interface.
- Memory Awareness and Optimized threading, for handle model.
- Continuous interactive user loop in Gradio Interface.
- Intelligent, context-aware response generation with summary.
- YAML state management for persistent settings, names and roles.
- VENV - It uses a virtual environment in a "./venv" folder.
- BASH - Bash Launcher-Installer for convinience. 

### Example Prompts
1) "Hello there! I never thought I would see you here on the mountain..."
2) "Wow, you can actually talk? What's your story?"
3) "You look wise, do you have any ancient wisdom to share?"
4) "Tell me, Wise-Llama, what is the purpose of humanity?"

### Requirements
- Ubuntu - Its programmed towards 24.04-24.10.

### USAGE (Windows)
1) Download and extract the package to a suitable folder, like "D:\Programs\Chat-VulkanLlama\", a path without spaces is always a better idea in general for GitHub projects.
2) Install dependencies via `Installer.Bat`, if you get errors, then either you, installed 7-Zip to an unusual folder, or otherwise you need to, turn your firewall off temporarely or make a rule to allow (I dont advise the second, as its for cmd).
3) Place a GGML model in `./models`, later on there will be selection from models library or something.. 
4) Launch with `Launcher.Bat`, by default, sounds and text to speach are ON, edit arguments in `Launcher.bat`. Optionally resize window with Ctrl + scroll mouse.

### USAGE (Linux):
- Download and use "LlmCppPyBot_v1p07", and some earlier ones, they possibly have linux compatibility, but I was never able to test this. So probably don't.

### CODE INFO:
- Main scripts: `window_1.py`, `window_2.py`.
- Utilities: `utility.py`, `model.py`, `interface.py`.

### TEST PROMPTS:
1) "Hello there! I never thought I would see you here on the mountain..."
2) "Wow, you can actually talk? What's your story?"
3) "You look wise, do you have any ancient wisdom to share?"
4) "Tell me, Wise-Llama, what is the purpose of humanity?"

### REQUIREMENTS:
- Windows 8-11 - The official Python 3.9 wont run on earlier versions of windows, but the batches will. Though if need be, then earlier versions of python can be installed/used see below.
- Python 3.9 non-Wsl (libraries `./data/requirements.txt`). Want to try other versions, then change "Python39" in the batch files to say "Python310", or whatever, in all cases. It will find it unless its in a wierd location.
- Large Language Models in Llama 3 GGUF format, I advise "L3-8B-Stheno-v3.2-GGUF-IQ-Imatrix", 8B is probably Optimal, the better q3s and most q4s, will be able to handle the text processing prompts.



## DISCLAIMER:
- Refer to License.Txt for terms covering usage, distribution, and modifications.
