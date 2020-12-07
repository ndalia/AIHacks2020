Prior to running any code, we recommend installing package requirements with "pip install -r requirements.txt". See "requirements.txt" for a list of requirements.

Any new data should be added to the provided csv files, specifically videos in the "video_watched_events_CONFIDENTIAL.csv" file and patients in the "patient_info_CONFIDENTIAL.csv". Using this data, we provide 2 notebooks for getting recommendations:
  
  - full.ipynb - run to both train and tests all models used for the categorizer and recommender systems. 
  
  - test.ipynb - run to test on saved pre-trained PCAs and NNs. Requires any new rows of data to be entered in "patient_info_CONFIDENTIAL.csv", "video_watched_events_CONFIDENTIAL.csv", and "videos_test.csv", which requires video categories to be represented as one-hot encodings (i.e. 1 for representative category, 0 for unrepresentative category).

While the test notebook requires additional data formatting (e.g. the categories must be well-defined), the overall runtime will be faster than when running both training and testing in the full notebook.

Both notebooks can be run either as Jupyter Notebooks or as python programs with their .py counterparts "full.py" and "test.py". Both the notebooks and the python scripts also require a value for a user id to get recommendations for the respective user, which can be entered from within the code. The python scripts can be easily run in a command line or terminal as "python full.py" and "python test.py"

