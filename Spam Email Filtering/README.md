# Spam Email Filtering
    Implemented a Multinomial Naive Bayes classifier for spam email filtering using a subset of Ling-Spam corpus to train and test the implemented system.

## Training and Test set
    Training and the test sets each contain 240 spam and 240 legitimate email messages.
    Each email message is provided as a separate file. All files start with a “subject:” heading.
    Stopword removal and lemmatization have already been performed.


Performed feature selection using the Document Frequency(DF) threshold approach seperately for the spam and legitimate classes(mails). 

DF threshold: The number of documents that contain the word.

Selected the top 200 words for spam class and the top 200 from the legitimate class and used these selections as features with the Multinomial Naive Bayes classifier.
