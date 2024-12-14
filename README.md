# Chat-Linux-Gguf
- Status: Working, see notes
- Note: Readme here is, mixed and missing, content, until, GPU and Image Generation, features are complete.

### Description
Its a lightweight Chatbot native to Linux, that uses Gguf models to simulate conversations with contextual awareness in a Gradio interface in your Web-Browser. The idea is its a framework able to be customised to adapt the model interference to any task or purpose, for example, personal manager, email responder, etc. 

### FEATURES
- Gguf Models: Using, Gguf-Parser and accompanying json files; auto-configuration of model parameters. 
- Gradio Interface: A browser-based interactive interface tied seamlessly into terminal operations.
- Integrated Setup and Operation: Through a single Bash launcher script that manages installation, execution, shutdown.
- Folder and File Management: Automated handling of configuration files, logs, and persistent data (YAML-based).
- 2 Prompt rotation for, conversation and consolidation, producing context aware conversation.
- Modularity: Python scripts are designed to work together with clear roles (e.g., model handling, interface logic).
- Persistence: Session data, settings, and configurations are retained across restarts.

### Preview
- The Conversation Page...

![preview_image](media/conversation.png)

- The Configuration Page...

![preview_image](media/configuration.png)

- The Installer/Launcher...
```
================================================================================
    Chat-Linux-Gguf - Bash Menu
================================================================================






    1. Launch Main Program

    2. Run Setup-Installer







--------------------------------------------------------------------------------
Selection; Menu Options = 1-2, Exit Program = X: 

```

### Requirements
- Linux OS - Modern Linux that is Ubuntu/Debian compatible.
- Python - Python is installed to `./VENV` does not affect system.
- LLMs - GGUF format Large Language Models, it will detect parameters.
- CPU (v1.00-v1.02) - Any x64 Cpu, scripts use standard llama-cpp-python. 
- GPU (=>v1.0? Upcoming) - nVidia CUDA Only, See notes for reasoning.
- Internet - Libraries and Compile stuff, bash installs from web.  

### Usage
Instructions are for upcoming GPU enhanced version...
1. Download the release version suted to your hardware, and then unpack to a sensible folder somewhere, then open a terminal in that folder, make "./Chat-Linux-Gguf.sh" executable.
1. In the terminal run the command "sudo ./Chat-Linux-Gguf.sh" in terminal in the program folder, and ensure the files are executable if there is immediate issue.
2. the Installer/Launcher runs, and you should select `2` to, install requirements and setup files/folders.
3. If you've multiple, GPUs and brands, a GPU menu will appear, select the GPU brand you intend to use.
3. After install completes, insert your `*.gguf` to the newly created folder `./models` (settings auto-detect).
4. You should then select `1` from the menu to launch the main program, and a browser window should pop-up, but in failing that then right click on `http://127.0.0.1:7860` and then left click `Open Link` to do the same.
5. You will then be interacting with the browser interface, where the buttons do what you would expect, but ensure to take a quick look at the "Configuration" tab first.
6. After finishing your session, then click on `Exit Program` in the browser window, and then the terminal will return to the Bash menu, and then select `X` to exit correctly. 

### Example Prompts
1) "Hello there! I never thought I would see you here on the mountain..."
2) "Wow, I bet you can actually talk? A talking llama, I'll be rich!"
3) "You look knowledgeable, do you have wise thoughts and think wise things?"
4) "Tell me, Wise-Llama, what is humanity's purpose, and dont tell me 42!"

### Notation
- Current, testing and advised, model(s): `https://huggingface.co/MaziyarPanahi/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF`.
- "Llama.cpp has been working on improving AMD GPU support, but the implementation is not as mature as NVIDIA's CUDA support." -Claude_Sonnet
- Originally designed for, 4 Python and 1 Bash, scripts; enabling editing with max 5 files in claude_sonnet for free account, thus, able to be edited by anyone.

### File Structure
- Initial File Structure...
```
./
├── Chat-Linux-Gguf.sh        # Main Bash launcher script
├── main_script.py             # Entry point script
├── requirements.txt           # Dependencies for the virtual environment
├── prompts/
│   ├── consolidate.txt        # Prompt template for consolidation tasks
│   ├── converse.txt           # Prompt template for conversation tasks
├── scripts/
│   ├── interface.py           # Gradio interface logic
│   ├── model.py               # GGUF model handling and interaction
│   ├── utility.py             # Utility functions
└── LICENSE.txt                # License file for the project
```
- Files Created by Installation...
```
./
├── data/
│   ├── temporary.py           # Holds ALL global variables
│   ├── __init__.py            # to mark the directory as a Python package
│   ├── persistent.yaml        # Holds default Chatbot configuration
├── venv/                      # Venv local install folder avoiding system conflict.
│   ├── *                      # Various libraries installed from `./requirements.txt`.
├── logs/                      # (Empty) Will contain any produced log files.
├── models/                    # (Empty) Directory for, `*.GGUF` and `model_config.json` ,files
```

 
### Development
The current plan for work featured is...
1. Image generation...
- Mostly implemented (see pictures).
- Testing models with more than 1 step, downloading 2/4 step versions of sdxl Lightning in Gguf.  
". Create plan for re-implementation of CUDA with Unified Memory, investigate how unified memory is done. 
- installer required to install, nvidia toolkit and nvidia cuda toolkit, then detect nvidia, `GPU Name`, `CUDA Cores Total`, `VRAM Total`. update hardware details to additionally detail the 3 new lines for the gpu.
- llama will be required to buid towards cuda, check compile arguments, use them to best effectiveness.
- see how best make use of knowing the number of cuda cores/vram, in the running of the main program; when using unified memory with the cuda compiled llama, what arguments are vaild? CPU Threads, Cuda Cores? whicever it is needs or both, needs to be there on hardware screen. 
- re-investigate test scripts. 
3. Optimize for Less Overall Characters, more advanced programming. 
4. Re-Structure Code/Scripts, ensure code is, appropriately and optimally, located in correctly themed/labeled script(s).
5. Test and Bugfix, all options and features, soes everything still works.

## DISCLAIMER:
- It is advided not to run the scripts in Alpha stage, or unexpected results may occur.
- Refer to License.Txt for terms covering usage, distribution, and modifications.
