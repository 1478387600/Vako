from LLM import *
from Speech_Recognition import *
from display import *

if __name__ == "__main__":

    add_display_line("Init automatic speech recognition...")
    asr_init()

    add_display_line("Init LLaMA GPT...")
    llm_init()

    while True:
        # Q-A loop:
        add_display_line("Start speaking")
        add_display_line("")
        question = transcribe_mic(chunk_length_s=5.0)
        if len(question) > 0:
            add_display_tokens(f"> {question}")
            add_display_line("")

            llm_start(question)
