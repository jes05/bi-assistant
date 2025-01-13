# bi-assistant
MLOPS project with multiple sample datasets to build a BI Assistant
The project takes multiple datasets from local or the database of the users choosing. 
Environment dependency - 
  Python -> requirements.txt to set up the virtual environment
  config.ini -> To save important sensitive information. 
Steps -> 
  ## 1. Feature Classification -
      The code automates identification of features automatically and then classifies them on the basis of whether the data is product, user, financial, logs etc as well as on the basis of datatypes.  
  ## 2. Interpreting Userquery to target the right datasets -
       Making use of logistic regression to target specific datasets that user requires output from to ensure user is taking the right output. Storing any new query in a separate file to ensure that the training happens smoothly and accuracy is maintained. 
