# Chat-Ubuntu-Gguf
- Status: Alpha

### Project Plans
The, conversion and re-engineer, of my WSL based ChatBot, to being Ubuntu 24 native...
1. Ensure Gradio interface has correct layout and configuration.
2. a Gradio Interface with model is loaded, so upon proper exit, ensure the, shutdown and unload, is comprihensive in the end_of_script function..
3. Continue with, Test and Bugfix, for all features/options, until stable.
4. Test, conversation and prompting, and examine responses, then improve.
3. Upon correct and working version, then Optimize for Less Overall Characters, more advanced programming.
2. Test and Bugfix, all options and features, soes everything still works.
4. Re-Structure Code/Scripts, ensure code is, appropriately and optimally, located in correctly themed/labeled script.
2. Test and Bugfix, all options and features, soes everything still works.
2. Release.
5. work on expansion of features, this will require a list of wanted features, then break down to, least code and best advantage, to round off features.
6. release final version.
8. update for new models as required.

### File Structure
- Initial File Structure...
```
./
├── Chat-Ubuntu-Gguf.sh        # Main Bash launcher script
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

### Description
Its a lightweight Chatbot native to Ubuntu 24, that uses Gguf models to simulate conversations with contextual awareness in a Gradio interface in your Web-Browser. There are 4 python scripts in, "./" and "./scripts", and 1 bash script in `./`, enabling editing with the maximum of 5 files upload in claude_sonnet for free account, thus, able to be modified into their own custom chatbot by anyone. The idea is its a framework able to be customised to adapt the model interference to any task or purpose, for example, personal manager, email responder, etc. Still being created, tba.  

### FEATURES
- Gguf Models: Using, Gguf-Parser and accompanying json files; auto-configuration of model parameters. 
- Gradio Interface: A browser-based interactive interface tied seamlessly into terminal operations.
- Integrated Setup and Operation: Through a single Bash launcher script that manages installation, execution, and cleanup.
- Folder and File Management: Automated handling of configuration files, logs, and persistent data (YAML-based).
- 2 Prompt rotation for, conversation and consolidation, producing context aware conversation.
- Modularity: Python scripts are designed to work together with clear roles (e.g., model handling, interface logic).
- Persistence: Session data, settings, and configurations are retained across restarts.
- Optimized for Ubuntu: Specifically tailored to Ubuntu 24.04–24.10 and AMD architecture.

### Preview
- Alpha Gradio Interface - Looking the part...

![preview_image](media/gradio_main.png)

- The Bash Installer/Launcher - under development...
```
================================================================================
    Chat-Ubuntu-Gguf
================================================================================




    1. Launch Main Program

    2. Run Setup-Installer




--------------------------------------------------------------------------------
Selection; Menu Options = 1-2, Exit Program = X: 
```

### Requirements
- Ubuntu - Its programmed on/towards Ubuntu 24.04-24.10.
- Python - It uses modern versions of Python in a VENV.
- LLMs - GGUF format with a provided, "model_config.json" or "config.json", file in same dir.
- AMD - Programmed on AMD, not currently testing on other platforms.

### Usage
- No working version verified yet.
1. When it works, it will run through "sudo ./Chat-Ubuntu-Gguf.sh" in terminal in the program folder.
2. the file "Chat-Ubuntu-Gguf.sh" is a Installer and Launcher, through menu; its already done mostly.
3. the python scripts search in `./models/` for the, `*.gguf` and `model_config.json`, files, and use that.
4. the user is presented with gradio interface popped up in default browser, and then have terminal somewhere.
5. the buttons do what you would expect, and the interface has all basic desired options available.

### Example Prompts
1) "Hello there! I never thought I would see you here on the mountain..."
2) "Wow, you can actually talk? What's your story?"
3) "You look wise, do you have any ancient wisdom to share?"
4) "Tell me, Wise-Llama, what is the purpose of humanity?"

### Notation
- Current, testing and advised, model(s): `https://huggingface.co/MaziyarPanahi/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF`.

## DISCLAIMER:
- It is advided not to run the scripts in Alpha stage, or unexpected results may occur.
- Refer to License.Txt for terms covering usage, distribution, and modifications.
