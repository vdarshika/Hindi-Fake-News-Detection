========================================================================================
Project : Anlysing performance  of  Bag-of-Word Related Features over Hindi Language for 
Fake News detectionsDevelope sequential Model for fake news detection for Hindi Language.
				(with LSTM and BERT)
========================================================================================

BY: 
Group : Grey Squad

FILES:

BERT.py
requirement.txt
final-datasets(folder for datasets)


INSTRUCTION----


REQUIREMENT: Refer requirement.txt file and install the modules if not already installed.


DATASET: Save all the data set(csv files) in a folder named "final-datasets" in the same folder.


TO RUN:

BERT-
	
	Following are the instruction needed to be followed on all the datasets

	1. Install the dependencies provided in the requirement file. 
	To run .py(python file) : open cmd and move to the current directory using cd.

   	Execute the following command
   	python BERT.py


	After EXECUTION: 

	1. Enter the name of the dataset for which you want to run the model for. first for train then for test. eg, "bbc_ner_train.csv".

	2. After feature extraction, model training and prediction, accuracy, class wise F-score and macro-average will be printed on the terminal.


LSTM-
	Following are the instruction needed to be followed on all the datasets

	1. Install the dependencies provided in the requirement file.
	2. To run .py(python file) : open cmd and move to the current directory using cd.
   		Execute the following command
   		python LSTM.py


	After EXECUTION: 

	1. Enter the name of the dataset for which you want to run the model for. first for train then for test. eg, "bbc_ner_train.csv".

	2. Trained model will be generated.
