# Music-Generation-using-Transformers-from-Scratch

This is an implementation of the paper "Attention is all you need".
Instead of using text, I tried to use sheet music notes as the tokens, and the best method is to use MIDI files.

To read the MIDI files, I used the code snippets from this site : "https://medium.com/analytics-vidhya/convert-midi-file-to-numpy-array-in-python-7d00531890c"

As for the transformer itself, is a standalone implementation of the paper mentioned, and works irrespective of the data being music or text.
I have created separate python files for the self and cross attention, encoder block, decoder block for future separate usage.

To use this, just replace the folder path to your folder path containing all music files in MIDI format.
