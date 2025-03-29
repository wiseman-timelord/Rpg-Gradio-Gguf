# Rpg-Gradio-Gguf (final name undecided)
- Status: Working.
- Note: readme.md here with, mixed and missing, content, until either, GPU and Image Generation, features are complete, but its >90% accurate. See development. Latest release for Text and Latest pre-release for Image generation also. Stuck in windows projects at the time of writing, must finish them first. Project also needs a cleanup and consistency check. download release versions.

### Description
Its a lightweight Chatbot native to Linux, that uses Gguf models to simulate conversations with contextual awareness in a Gradio interface in your Web-Browser. The idea is its a framework able to be customised to adapt the model interference to any task or purpose, for example, personal manager, email responder, etc. 

### FEATURES
- Gguf Models: Using, Gguf-Parser and accompanying json files; auto-configuration of model parameters. 
- Gradio Interface: A browser-based interactive interface tied seamlessly into terminal operations.
- Integrated Setup and Operation: Through a single Bash launcher script that manages installation, execution, shutdown.
- Folder and File Management: Automated handling of configuration files, logs, and persistent data (YAML-based).
- 3 Prompt rotation for, conversation and consolidation and image generation, producing context aware experience.
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
    Rpg-Gradio-Gguf - Bash Menu
================================================================================






    1. Launch Main Program

    2. Run Setup-Installer







--------------------------------------------------------------------------------
Selection; Menu Options = 1-2, Exit Program = X: 

```

### Requirements
- Linux OS - Modern Linux that is Ubuntu/Debian compatible.
- Python - Python is installed to `./VENV` does not affect system.
- LLMs - Advised models, SDXL-Lightning and Llama3.1, in Gguf. 
- CPU (v1.00-v1.02) - Any x64 Cpu, scripts use standard llama-cpp-python. 
- GPU (=>v1.04 Upcoming) - nVidia CUDA Only, See notes for reasoning.
- Internet - Libraries and Compile stuff, bash installs from web.  

### Usage
Instructions are for upcoming GPU enhanced version...
1. Download the release version suted to your hardware, and then unpack to a sensible folder somewhere, then open a terminal in that folder, make "./Chat-Linux-Gguf.sh" executable.
1. Ensure the files are executable by right clicking them and ticking the appropriate box, or alternatively running `sudo chmod -x Rpg-Gradio-Gguf.sh`.
2. In the terminal run the command "sudo ./Chat-Linux-Gguf.sh", its a installer/launcher, libraries install to `./venv`. 
2. the Installer/Launcher runs, and you should select `2` to, install requirements and setup files/folders, it will then return to the menu.
3. (not currently applicable) If you've multiple, GPUs and brands, a GPU menu will appear, select the GPU brand you intend to use.
3. After install completes, insert your, text GGUF to `./data/text` and image GGUF to `./data/image`, the settings will auto-detect.
4. You should then select `1` from the menu to launch the main program, and a browser window should pop-up, or if not then right click on `http://127.0.0.1:7860` and then `Open Link`.
5. You will then be interacting with the browser interface, where the buttons do what you would expect, but ensure to take a quick look at the "Configuration" tab first.
6. After finishing your session, then click on `Exit Program` in the browser window, and then the terminal will return to the Bash menu, and then shutdown correctly. 

### Example Prompts
1) "Hello there! I never thought I would see you here on the mountain..."
2) "Wow, I bet you can actually talk? A talking llama, I'll be rich!"
3) "You look knowledgeable, do you have wise thoughts and think wise things?"
4) "Tell me, Wise-Llama, what is humanity's purpose, and dont tell me 42!"

### Notation
- Current advised text model(s): `https://huggingface.co/MaziyarPanahi/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF` in Q8_0; this model works.
- New model for NSFW text, amazing [Llama 3.1 NSFW GGUF](https://huggingface.co/Novaciano/Llama-3.2-3b-NSFW_Aesir_Uncensored-GGUF)
- New highly compitent NSFW model (though maybe slow by comparrison to others) `etr1o-v1.2-GGUF`
- [flux v2 nsfw GGUF](https://huggingface.co/Anibaaal/Flux-Fusion-V2-4step-merge-gguf-nf4) - able to complete image generation in 4 steps. Try this.
- Current advised image model(s): *unknown* in Q8_0 (when I find one that works correctly, then..).
- "Llama.cpp has been working on improving AMD GPU support, but the implementation is not as mature as NVIDIA's CUDA support." -Claude_Sonnet
- Originally designed for, 4 Python and 1 Bash, scripts; enabling editing with max 5 files in claude_sonnet for free account, thus, able to be edited by anyone.


### File Structure
- Initial File Structure...
```
./
├── Rpg-Gradio-Gguf.sh        # Main Bash launcher script
├── main_script.py             # Entry point script
├── data/ 
│   ├── new_session.jpg        # Default Image
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
│   ├── __init__.py            # to mark the directory as a Python package
│   ├── persistent.yaml        # Holds default Chatbot configuration
│   ├── requirements.txt       # Dependencies for the virtual environment
│   ├── temporary.py           # Holds ALL global variables
├── venv/                      # Venv local install folder avoiding system conflict.
│   ├── *                      # Various libraries installed from `./data/requirements.txt`.
├── logs/                      # (Empty) Will contain any produced log files.
├── models/                    # (Empty) Directory for, `*.GGUF` and `model_config.json` ,files
```

 
### Development
The current plan for work featured is...
1. Splitting install processes into its own python script, to make more compitent processes, in mind of new requirements to install pre-compiled binaries. Bash then becomes able to be run without sudo unless running installer from menu. Needs printed line to alert user of this on menu.
1. Image generation (1.03)...
- It was implemented, but then stable diffusion cpp python cannot load the newer stable diffusion >3.5-Turbo models in a gguf format, this is important because sdxl-lightning not able to do it in 4 steps, where as 3.5 can. Solution is to use llama-box, this tool will be able to load both, text and image, models; re-implementing interference method.
- Testing models are now, sd3.5 large/large_turbo; `https://huggingface.co/gpustack/stable-diffusion-v3-5-large-GGUF/tree/main` and `https://huggingface.co/gpustack/stable-diffusion-v3-5-large-turbo-GGUF/tree/main`.
2. Create plan for re-implementation of CUDA with Unified Memory, investigate how unified memory is done. 
- installer required to install, nvidia toolkit and nvidia cuda toolkit, then detect nvidia, `GPU Name`, `CUDA Cores Total`, `VRAM Total`. update hardware details to additionally detail the 3 new lines for the gpu.
- llama will be required to buid towards cuda, check compile arguments, use them to best effectiveness.
- see how best make use of knowing the number of cuda cores/vram, in the running of the main program; when using unified memory with the cuda compiled llama, what arguments are vaild? CPU Threads, Cuda Cores? whicever it is needs or both, needs to be there on hardware screen. 
- re-investigate test scripts. 
3. text boxes need output/input to be like `{human_name}: Message contents here` or `{agent_name}: Message contents here`.
3. Optimize for Less Overall Characters, more advanced programming. 
4. Re-Structure Code/Scripts, ensure code is, appropriately and optimally, located in correctly themed/labeled script(s).
5. Test and Bugfix, all options and features, soes everything still works.

## DISCLAIMER:
- It is advided not to run the scripts in Alpha stage, or unexpected results may occur.
- Refer to License.Txt for terms covering usage, distribution, and modifications.
