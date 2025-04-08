import pandas as pd
from convokit import Corpus, download
import os

def get_id(utterance):
    """
    Extract the numeric ID from an utterance ID.
    
    Args:
        utterance: The utterance object
        
    Returns:
        int: The numeric ID extracted from the utterance ID
    """
    return int(utterance.id[1:])

def get_convo_id(utterance):
    """
    Get the conversation ID from an utterance.
    
    Args:
        utterance: The utterance object
        
    Returns:
        str: The conversation ID
    """
    return utterance.conversation_id

def get_conversations_dataframe(corpus, num_convos = None, selector=lambda convo: True, exclude_meta: bool = True):
    """
    Get a DataFrame of the conversations from a corpus.
    
    This function extracts conversation data from a corpus and returns it as a DataFrame.
    It can filter conversations using a selector function and optionally exclude metadata.
    
    Args:
        corpus: The corpus to extract conversations from
        num_convos (int, optional): Maximum number of conversations to include. Defaults to None.
        selector (function, optional): Function to filter conversations. Defaults to lambda convo: True.
        exclude_meta (bool, optional): Whether to exclude metadata. Defaults to True.
        
    Returns:
        pandas.DataFrame: DataFrame containing conversation data
    """
    ds = dict()
    i = 0
    for convo in corpus.iter_conversations(selector):
        if num_convos is not None and i >= num_convos:
            break
        d = convo.to_dict().copy()
        if not exclude_meta:
            for k, v in d["meta"].items():
                d["meta." + k] = v
        del d["meta"]
        ds[convo.id] = d

    df = pd.DataFrame(ds).T
    return df.set_index("id")

def get_corpus_df(num_convos = None):
    """
    Get the corpus and DataFrame, with caching.
    
    This function downloads or loads the movie corpus and returns it as a DataFrame.
    It caches the corpus to avoid downloading it multiple times.
    
    Args:
        num_convos (int, optional): Maximum number of conversations to include. Defaults to None.
        
    Returns:
        tuple: (corpus, df) - The corpus and DataFrame
    """
    # Define cache directory
    cache_dir = os.path.expanduser("~/.convokit/saved-corpora")
    corpus_path = os.path.join(cache_dir, "movie-corpus")
    if not os.path.exists("movie_corpus.csv") or not os.path.exists(corpus_path):  
        
        # Check if corpus exists in cache
        if os.path.exists(corpus_path):
            print("Loading corpus from cache...")
            corpus = Corpus(filename=corpus_path)
        else:
            print("Downloading corpus (this will only happen once)...")
            corpus = Corpus(filename=download("movie-corpus"))
        
        df = get_conversations_dataframe(corpus, num_convos)
        df.to_csv("movie_corpus.csv")
        
    
    else:
        df = pd.read_csv("movie_corpus.csv")
        corpus = Corpus(filename=corpus_path)

    return corpus, df

def get_dictionary_of_movies(N_MOVIES_TO_USE=300, N_CONVOS = None, start_movie_id=0):
    """
    Get a dictionary of movies with their text and speaker IDs.
    
    This function extracts movie dialogue and speaker IDs from the corpus.
    It returns a dictionary with two lists: 'speaker_ids' and 'text'.
    
    Args:
        N_MOVIES_TO_USE (int, optional): Number of movies to include. Defaults to 300.
        N_CONVOS (int, optional): Maximum number of conversations per movie. Defaults to None.
        start_movie_id (int, optional): Starting movie ID. Defaults to 0.
        
    Returns:
        dict: Dictionary containing 'speaker_ids' and 'text' lists
    """
    corpus, df = get_corpus_df(N_CONVOS)
    utterance_ids = corpus.utterances.keys()

    all_movies = {"speaker_ids":[],"text":[]}
    for i in range(start_movie_id, start_movie_id + N_MOVIES_TO_USE):
        utterance_in = [corpus.utterances[u] for u in utterance_ids if corpus.utterances[u].meta.__getitem__('movie_id') == f"m{i}"]
        utterance_in.sort(key=get_id)
        speaker_ids = [u.speaker.id for u in utterance_in]
        line_ids = [u.id for u in utterance_in]
        text = [u.text for u in utterance_in]
        all_movies['speaker_ids'].append(speaker_ids)
        all_movies['text'].append(text)

    return all_movies