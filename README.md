# caramba
[C]haracter [A]wa[r]e Screenplay Development with J[AMBA]

### Preprocessing/Data
The dataset that gets fed into training should have a single entry per movie in the Cornell dataset. Each entry should have attributes input_ids and token_speaker_ids, which respectively show the input tokens of every word in the script and the speaker id attached to each token. To obtain this dataset, you can reshape the original Cornell Movie database with reshape_movies.py (to obtain a dictionary for which each movie is an element), and then a Torch tokenized dataset with dataset.py.
