# Content-Moderation-Tool
Content moderation tool on demo Twitter web application. Uses BERT to predict semantics behind a tweet (hateful or non-hateful). 

You will need to run the application locally because the weights.pt file (the set of of parameters that allows for new inputs to be correctly classified) is too large and cannot be pushed into this repository. 

1. Download project onto your device.
3. First, you need to train the BERT model. Simply run the train.py file.
4. Training can take up to several hours. At the end, once testing is conducted, a classification report should be provided stating the precision and accuracy. 
5. A weights.pt file will be saved into the same directory as the train.py file "project/main". Move that file into the parent directory.
6. Then in terminal, go to the project directory and type "python server.py" to run flask server.
7. You will be given a link to click and test out the tool.


NOTE:
Make sure all the necessary packages have been installed. You can find out how to install here: https://packaging.python.org/en/latest/tutorials/installing-packages/ 

Here is the list of all packages needed: 
flask
numpy
pandas
torch
transformers
scikit-learn
matplotlib
imbalanced-learn

