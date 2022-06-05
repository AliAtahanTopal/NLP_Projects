# Abstract Stemmer

To make the executable file I used py2exe libraries. If you don't have it installed you can install in terminal

-> pip install py2exe 

After installing py2exe I created a file named setup.py to make the Assignment2.py a executable. I opened a terminal from where the setup.py and assignment2.py are and wrote in order

-> python setup.py install

-> python setup.py py2exe 

the last command created a folder named "dist". The Assignment2.exe is inside the dist folder. But to run the executable you have to add the corpus.txt, test-misspelled-words.txt and test-words-correct.txt to the dist folder. After adding the txt filed to the dist folder I moved my terminal to the dist folder with command

-> cd dist 

and tried if Assignment2.exe works as intended. 

-> Assignment2.exe

There were no errors so the executable worked. After each run there will be two new text files that are named 'corrected_words.txt' for the non enhanced spell corrected words and 'corrected_words_enhanced.txt' for the enhanced version of the first algorithm. I would recommend you to delete these two newly created files because in every run it will write the new words after the old words. If you want to see after using 2-3 text files it will keep every word that is corrected. It can affect the accuracy score. Lets say that the first misspelled word txt file has 180 and the second has 180 as well after running these two without deleting any file the output files will have 360 words in them. 

There is a input section in the program. You can use that place to try one word for a time to be corrected without writing to any file.

REMEMBER TO PUT THE CORUS.TXT, TEST_WORDS_MISPELLED.TXT and TEST_WORDS_CORRECT.TXT FILES TO THE 'dist' FOLDER BEFOURE YOU RUN. THE EXECUTABLE IS INSIDE THE 'dist' FOLDER AS WELL.
