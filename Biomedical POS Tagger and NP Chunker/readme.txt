HMM POS Tagger & Noun Phrases Extractor
------------------------------------------
You can run the script via cmd/anaconda_prompt or an IDE like Spyder.
To run via cmd: "python path_to_file/nlp.py"
------------------------------------------
Guide:
1. When you run the script, it will start to read and train the data (it can take upto 5 mins)
2. Current process of operations are shown in output with percentages.
3. After training completed. Program will show a menu for you to choose what you want to do. It will look like this:
	
	Model is trained and ready for action.
	=====================================
	Enter 
	'1' for tagging the genia-test set.
	'2' for tagging a custom .txt file.
	'0' to exit.
	Waiting for input:

4. 	
	If you choose 1 it will test the model on genia-test file. It will show you the accuracy, write the POS Tagger output to the "genia-tags.txt" file and write the noun phrases found to the "genia-noun-phrases.txt"
	
	If you choose 2, it will show you another menu for format selection and it will look like this:

	Choose .txt format:
	1: Regular text
	2: Genia format with tags
	3: Lined format without tags

		1-> Written as usual text file.
		2-> Formatted exactly like genia-test.txt
		3-> Words are in different lines
		You can see examples of formats in project folder (format1,format2,format3)

			After you select one of the formats it will ask you to enter the .txt filename. Remember you should only enter the name without extension (.txt). For example if your file is "example.txt" you should enter "example" without quotes.
			!!! IMPORTANT !!!: the .txt file you are going to enter should be in the same directory with "nlp.py" file.

				After you enter the filename, program will read the text, tag the words and extract the noun phrases. Tag outputs will be written to the "custom_tags.txt" file, noun phrases will be written to "custom_noun_phrases.txt" file.

	If you choose 0. Program will be terminated.

------------------------------------------
Project Files:
	nlp.py: Source code of the project.
	genia-train.txt: Genia train-set
	genia-test.txt: Genia test-set
	genia-noun-phrases.txt: Noun Phrases function output for genia test set.
	genia-tags.txt: POS Tagger output for genia test set.
	custom-noun-phrases.txt: Noun Phrases function output file for custom sets.
	custom_tags.txt: POS Tagger output file for custom sets.
	format1.txt: Example text for format1
	format2.txt: Example text for format2
	format3.txt: Example text for format3
------------------------------------------
Ali Atahan Topal