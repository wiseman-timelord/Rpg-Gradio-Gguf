# .\data\params\temporary.py

# General Variables
session_history = "the conversation started"  # Default: "the conversation started"
rotation_counter = 0

# Model Variables
loaded_models = {}
llm = None

# Configurable Keys
agent_name = "Empty"
agent_role = "Empty"
human_name = "Empty"

# Other Keys
session_history = "the conversation started"
agent_output = ""
human_input = ""

# Model Mapping
MODE_TO_TEMPERATURE = {
    'RolePlaying': 0.7,
    'TextProcessing': 0.1
}

PROMPT_TO_MAXTOKENS = {
    'converse': 2000,
    'consolidate': 1000
}

# Syntax Options
SYNTAX_OPTIONS_DISPLAY = [
    "{combined_input}",
    "User: {combined_input}",
    "User:\\n{combined_input}",
    "### Human: {combined_input}",
    "### Human:\\n{combined_input}",
    "### Instruction: {combined_input}",
    "### Instruction:\\n{combined_input}",
    "{system_input}. USER: {instruct_input}",
    "{system_input}\\nUser: {instruct_input}"
]
SYNTAX_OPTIONS = SYNTAX_OPTIONS_DISPLAY

# Paths
RAMFS_DIR = '/mnt/ramfs'
PERSISTENT_FILE = f"{RAMFS_DIR}/persistent.yaml"

