[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/tISOEztM)
# Assignment 3  - distributed in GitHub Repo e4040-2024Fall-assign3

The assignment is distributed as several Jupyter notebooks and a number of directories and subdirectories in utils.

# Detailed instructions on how to submit this assignment/homework:

1. The assignment will be distributed as a Github classroom assignment - as a special repository accessed through a link
2. A student's copy of the assignment gets created automatically with a special name - <span style="color:red"><strong>students must rename per instructions below</strong></span>
3. The solution(s) to the assignment have to be submitted inside that repository as a set of "solved" Jupyter Notebooks, and several modified python files which reside in directories/subdirectories. This step is done by pushing all the files to the main branch.
4. Three files/screenshots need to be uploaded into the directory `figures/` which prove that the assignment has been done in the cloud.
5. Submit your repository through Gradescope.


## (Re)naming of the student repository (TODO students) 

INSTRUCTIONS for naming the student's solution repository for assignments with one student:
* This step will require changing the repository name
* Students MUST use the following name for the repository with their solutions: e4040-2024Fall-assign3-UNI (the first part "e4040-2024Fall-assign3" will probably be inherited from the assignment, so only UNI needs to be added) 
* Initially, the system will give the repo a name that ends with a  student's Github userid. The student MUST change that name and replace it with the name requested in the point above
* Good Example: e4040-2024Fall-assign3-zz9999;   Bad example: e4040-2024Fall-assign3-e4040-2024Fall-assign3-zz9999.
* This change can be done from the "Settings" tab which is located on the repo page.

INSTRUCTIONS for naming the students' solution repository for assignments with more students, such as the final project. Students need to use a 4-letter groupID): 
* Template: e4040-2024Fall-Project-GroupID-UNI1-UNI2-UNI3. -> Example: e4040-2024Fall-Project-MEME-zz9999-aa9999-aa0000.


# Organization of this directory

TODO students: Run commands to create a directory tree, as described in previous assignments

```   

├── Assignment3-intro.ipynb
├── README.md
├── img
│   ├── bptt.png
│   ├── bptt2.jpg
│   ├── charrnn.jpg
│   ├── doubleLSTM.png
│   ├── embedding.png
│   ├── lookback.png
│   ├── lstm_cell.png
│   ├── prediction.png
│   ├── seq2seq-inference.png
│   ├── seq2seq-teacher-forcing.png
│   ├── seq2seq.jpg
│   ├── singleLSTM.png
│   ├── tsne_female_male.png
│   ├── xnor.png
│   └── xor.png
├── requirements.txt
├── stock_data
│   └── Microsoft_Stock.csv
├── task1-xor.ipynb
├── task2-rnn-forecasting.ipynb
├── task3-rnn-translation.ipynb
├── text_data
│   ├── eng_vocab.txt
│   ├── nl_vocab.txt
│   ├── nmt_eng.npy
│   └── nmt_nl.npy
└── utils
    ├── LSTM.py
    ├── __pycache__
    │   ├── LSTM.cpython-36.pyc
    │   └── dataset.cpython-36.pyc
    ├── dataset.py
    └── translation
        ├── __pycache__
        │   ├── layers.cpython-36.pyc
        │   ├── text_data.cpython-36.pyc
        │   └── train_funcs.cpython-36.pyc
        ├── layers.py
        ├── text_data.py
        └── train_funcs.py
        
7 directories, 36 files
