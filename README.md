# Content-Moderation-Tool
Content moderation tool on demo Twitter web application. Uses BERT to predict semantics behind a tweet (hateful or non-hateful). 

You will need to run the application locally because the weights.pt file (the set of of parameters that allows for new inputs to be correctly classified) is too large and cannot be pushed into this repository. 

1. Download project onto your device
2. First, you need to train the BERT model. Simply run the train.py file.
3. Training can take up to several hours. At the end, once testing is conducted, a classification report should be provided stating the precision and accuracy. 
4. A weights.pt file will be saved into the same directory as the train.py file "project/main". Move that file into the parent directory.
5. Then in terminal, go to the project directory and type "python server.py" to run flask server.
6. You will be given a link to click and test out the tool.


NOTE:
Make sure all the necessary libraries have been installed. 

