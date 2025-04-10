#! /bin/bash

python3 longformer_token_prediction.py --model "allenai/longformer-base-4096" --max_length 300 --local_window 100 --num_movies 10 --start_movie_id 0



