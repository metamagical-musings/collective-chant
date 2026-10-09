This directory contains the two files whisper_cli.py and whisper_tools.py.

They were created with the help of AI and are useful for testing and tuning
faster-whisper (or whisper, with a few modifications), which is a Python module
for automatic speech recognition (ASR).

They are configured to split a voice recording into stanzas, each less than
30 seconds in duration, separated by at least 1 second, to take advantage
of whisper's better performance during the first 30 seconds.
