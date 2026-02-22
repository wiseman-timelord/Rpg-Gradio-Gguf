# Rpg-Gradio-Gguf
- Status: Project restart.

### Description
Its a Chatbot with, text and image, generation, tuned to RPG, that uses Gguf models to simulate limitless roleplaing in a Gradio interface in Web-Browser interface. 

### FEATURES
- Gguf Models: Compressed large language models with auto-configuration of model parameters. 
- Gradio Interface: A browser-based interactive interface tied seamlessly into terminal operations.
- Integrated Setup and Operation: Through a single Batch launcher script that manages installation and execution.
- 3 Prompt rotation for, conversation and consolidation and image generation, producing context aware experience.
- Persistence: Session data, settings, and configurations are retained across restarts.

### Preview
- The Conversation Page...

![preview_image](media/conversation.png)

- The Configuration Page...

![preview_image](media/configuration.png)

- The Installer/Launcher...
```
================================================================================
    Rpg-Gradio-Gguf - Bash Menu
================================================================================






    1. Launch Main Program

    2. Run Setup-Installer







--------------------------------------------------------------------------------
Selection; Menu Options = 1-2, Exit Program = X: 

```

### Requirements
- Windows - Without proper assessment of the scripts, its Windows 10.
- Python - Without proper assessment of the scripts, its Python 3.12.
- LLMs - The model we are using is [Wan2.1_T2V_14B_FusionX-GGUF](https://huggingface.co/QuantStack/Wan2.1_T2V_14B_FusionX-GGUF/tree/main)
- CPU - Any x64 Cpu, scripts use standard llama-cpp-python. 
- GPU - Vulkan capable GPU Only, it uses vulkan.
- Internet - Installer requires internet, main program will be offline.  

### Usage
Instructions are for upcoming GPU enhanced version...
```
t.b.a
```

### Notation
- T.B.A.


### File Structure
- Initial File Structure...
```
./
├── Rpg-Gradio-Gguf.bat        # Main Batch entry script
├── main_script.py             # Entry point to program, startup/shutdown functions, main loop
├── data/ 
│   ├── new_session.jpg        # Default Image
├── scripts/
│   ├── configure.py           # globals/maps/lists, load/save json functions
│   ├── displays.py           # Gradio interface, browser code
│   ├── inference.py               # GGUF model handling, model prompting
│   ├── utilities.py             # Utility functions
└── LICENSE.txt                # License file for the project
```
- Files Created by Installation...
```
./
├── data/
│   ├── __init__.py            # to mark the directory as a Python package
│   ├── persistent.json        # Holds default Chatbot configuration
├── venv/                      # Venv local install folder avoiding system conflict.
│   ├── *                      # Various libraries installed from `./data/requirements.txt`.
├── logs/                      # (Empty) Will contain any produced log files.
├── models/                    # (Empty) Directory for, `*.GGUF` and `model_config.json` ,files
```

 
### Development
The current plan for work featured is...
1. Restart and complete.

## DISCLAIMER:
- It is advided not to run the scripts in Alpha stage, or unexpected results may occur.
- Refer to License.Txt for terms covering usage, distribution, and modifications.
