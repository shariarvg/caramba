import pandas as pd
from convokit import Corpus, download

def get_id(utterance):
        return int(utterance.id[1:])

def get_convo_id(utterance):
    return utterance.conversation_id

def get_corpus_df():
    corpus = Corpus(filename=download("movie-corpus"))
    df = corpus.get_conversations_dataframe()
    return corpus, df

def get_dictionary_of_movies(N_MOVIES_TO_USE=300):
    corpus, df = get_corpus_df()
    utterance_ids = corpus.utterances.keys()

    all_movies = {"speaker_ids":[],"text":[]}
    for i in range(N_MOVIES_TO_USE):
        utterance_in = [corpus.utterances[u] for u in utterance_ids if corpus.utterances[u].meta.__getitem__('movie_id') == f"m{i}"]
        utterance_in.sort(key=get_id)
        speaker_ids = [u.speaker.id for u in utterance_in]
        line_ids = [u.id for u in utterance_in]
        text = [u.text for u in utterance_in]
        all_movies['speaker_ids'].append(speaker_ids)
        all_movies['text'].append(text)

    return all_movies