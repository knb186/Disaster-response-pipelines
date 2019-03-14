# Disaster-response-pipelines

The disaster response pipelines project aims to take translated Twitter messages from a storm and analyze them depending on which categories they belong to. This data is used to train a machine learning pipeline that can be used to take future messages and give a 0 or 1 response on each of the categories. 

In the repository, the ETL Pipeline file cleans the original dataset with error messages. The process_data.py produces the same result but does so in a way that allows for another file to be used in the terminal. The ML Pipeline file uses a cleaned dataset to construct a machine-learning model. Similarly the train_classifier.py file can be used to insert another dataset in the terminal than the one used in the Jupyter notebook file. The run.py file allows for the model to run and display visualizations. The categories.csv and messages.csv are the datasets that were used in the ETL Pipeline/ML Pipeline files.
