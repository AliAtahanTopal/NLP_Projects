# Importing libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys

os.chdir(os.path.dirname(sys.argv[0]))


def split_words_tags(content):
    word_tags = [('START',
                  'START')]  # Defining a tag to show a sentence is starting. For the first sentence there is no = signs or START tag at first.
    for i in tqdm(range(len(content) - 1)):
        if content[i] == '.====================':
            word_tags.append(('.', '.'))
            word_tags.append(('END', 'END'))
            word_tags.append(('START', 'START'))
        elif content[i].split('/')[
            0] == '====================':  # Changing the end and start word because there was no tag for this.
            if i == len(content) - 2:
                word_tags.append(('END', 'END'))
            else:
                word_tags.append(('END', 'END'))
                word_tags.append(('START', 'START'))
        else:
            word_tags.append((content[i].split("/")[-1], '/'.join(content[i].split("/")[:-1])))

    return word_tags


def create_set(tag_words):
    sentences = []
    sentence_array = []
    for line in tqdm(tag_words):
        if 'START' in line[0] or ():
            continue
        elif 'END' in line[0]:
            sentences.append(sentence_array)
            sentence_array = []
        else:
            new_line = (line[1], line[0])
            sentence_array.append(new_line)

    return sentences


def split_sentences(content):
    sentences_w_tags = []
    for i in range(len(content)):
        sentences_w_tags.append(content[i].split("\n"))

    sentences = []
    for j in range(len(sentences_w_tags)):
        sentence = sentences_w_tags[j]
        for wt in sentence:
            word = '/'.join(wt.split("/")[:-1])
            sentences.append(word)

    return sentences


def t2_given_t1(t2, t1, train_bag):
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t == t1])
    count_t2_t1 = 0
    for index in range(len(tags) - 1):
        if tags[index] == t1 and tags[index + 1] == t2:
            count_t2_t1 += 1
    return count_t2_t1, count_t1


def word_likelihood(words, tags):
    unique_words = list(set(words))
    unique_tags = list(tags)
    columns = ['words']
    columns = np.append(columns, unique_tags)
    df = pd.DataFrame(columns=columns)
    df['words'] = unique_words
    df = df.set_index('words', drop=True)
    df = df.fillna(0)
    for line in tqdm(train_tag_words):
        if ('START' in line) or ('END' in line):
            continue
        else:
            t = line[0]
            w = line[1]
            df.loc[w, t] += 1
    return df


def viterbi(sentence, tags, WLP, TT, train_words):
    state = []

    for i in range(len(sentence)):
        m = -1
        word = sentence[i]
        p = []
        for tag in tags:
            if i == 0:
                transition_p = 1
            else:
                transition_p = TT.loc[state[-1], tag]

            if word in train_words:
                likelihood_p = WLP.loc[word, tag]
            else:
                likelihood_p = 1
            state_p = likelihood_p * transition_p

            if state_p > m:
                m = state_p
            p.append(state_p)

        state_max = tags[p.index(m)]
        state.append(state_max)

    return list(zip(sentence, state))


def split_sentences2(content):
    sentences_w_tags = []
    for i in range(len(content)):
        sentences_w_tags.append(content[i].split("\n"))

    sentences = []
    for j in range(len(sentences_w_tags)):
        sentence = sentences_w_tags[j]
        sentence2 = []
        for wt in sentence:
            word = '/'.join(wt.split("/")[:-1])
            sentence2.append(word)

        sentences.append(sentence2[:-1])

    return sentences


def noun_phrases(viterbi_results):
    nps = []
    for i in tqdm(viterbi_results):
        np_word_tags = []
        for j in i:
            if j[-1] in ['dt', 'jj', 'jjr', 'jjs', 'nn', 'nns', 'nnp', 'nnps', 'prp', 'prp$']:
                if j[-1] == 'dt':
                    if len(np_word_tags) == 0:
                        np_word_tags.append(j)
                if (j[-1] == 'jj') or (j[-1] == 'jjr') or (j[-1] == 'jjs'):
                    np_word_tags.append(j)
                if (j[-1] == 'nn') or (j[-1] == 'nns') or (j[-1] == 'nnp') or (j[-1] == 'nnps') or (j[-1] == 'prp') or (
                        j[-1] == 'prp$'):
                    if (len(np_word_tags) != 0) and (
                            (np_word_tags[-1][-1] == 'dt') or (np_word_tags[-1][-1] == 'jj') or (
                            np_word_tags[-1][-1] == 'jjr') or (np_word_tags[-1][-1] == 'jjs')):
                        np_word_tags.append(j)
                        nps.append(np_word_tags)
                        np_word_tags = []

                if (j[-1] == 'nnp') or (j[-1] == 'nnps') or (j[-1] == 'prp') or (j[-1] == 'prp$'):
                    np_word_tags.append(j)
                    nps.append(np_word_tags)
                    np_word_tags = []
            else:
                np_word_tags = []

    return nps


def split_sentences3(content):
    content = content.lower().replace('.', ' .').replace('?', ' ?').replace(':', ' :').replace(')', ' )').replace('(',
                                                                                                                  '( ').replace(
        '!', ' !').replace(',', ' ,').replace('_', ' _')
    content = content.split(' ')
    splitted_words = []
    a = []
    for i in content:
        if i[0] == '"':
            a.append('"')
            a.append(i[1:-1])
            a.append('"')
            continue
        if (i == '.') or (i == '?') or (i == '!'):
            a.append(i)
            splitted_words.append(a)
            a = []
        else:
            a.append(i)

    return splitted_words


def split_sentences4(content):
    sentences = content.split("\n")
    sent_arr = []
    arr = []
    for i in range(len(sentences)):
        if sentences[i] != '.':
            arr.append(sentences[i])
        else:
            arr.append(sentences[i])
            sent_arr.append(arr)
            arr = []
    return sent_arr


def convert_2_string(sentences_array):
    out_string = ""
    for sentence in sentences_array:
        for word in sentence:
            out_string += word

    return out_string


def test_genia():
    print("Please wait. It will take approximately 30 minutes.")
    test_string = open(test_path).read().lower().split("====================\n")
    test_sentences = split_sentences2(test_string)

    viterbi_results = []
    for sentence in tqdm(test_sentences):
        viterbi_results.append(viterbi(sentence, distinct_train_tags, WLP, TT, distinct_train_words))

    print("--> POS Tagging completed for\n", len(viterbi_results), "sentences and", len(test_tagged_words), "words.")
    count = 0
    t_count = 0
    for j in range(len(viterbi_results) - 1):
        for t in range(len(viterbi_results[j])):
            t_count += 1
            x1 = test_set[j][t]
            x2 = viterbi_results[j][t]
            if x1 == x2:
                count += 1

    print("--> with ", (count / t_count) * 100, "%", " accuracy")
    print("Saving the results...")
    write_tags_out(viterbi_results, 'genia-tags')
    print("Tagging result is saved as 'genia-tags.txt' in the script's directory.\n")
    print("Extracting the Noun Phrases")
    nps = noun_phrases(viterbi_results)
    print("--> Extraction completed.\n", len(nps), "noun phrases found.")
    print("Saving the noun phrases...")
    write_np_out(nps, 'genia-noun-phrases')
    print("Noun phrases are saved as 'genia-noun-phrases.txt' in the script's directory.\n")


def test_custom1(text_name):
    print("\n=============================\n")
    print("Reading the txt file...\n")
    string = open(text_name).read().lower()
    a = split_sentences3(string)
    print("Running viterbi algorithm... (this may take a long time depending on the test size)\n")
    viterbi_results_a = []
    for sentence in tqdm(a):
        viterbi_results_a.append(viterbi(sentence, distinct_train_tags, WLP, TT, distinct_train_words))

    print("--> POS Tagging completed for", len(viterbi_results_a), "sentences.")
    print("Saving the results...")
    write_tags_out(viterbi_results_a, 'custom_tags')
    print("Tagging result is saved as 'custom_tags.txt' in the script's directory.\n")
    print("Extracting the Noun Phrases")
    np_a = noun_phrases(viterbi_results_a)
    print("--> Extraction completed.\n", len(np_a), "noun phrases found.")
    print("Saving the noun phrases...")
    write_np_out(np_a, 'custom_noun_phrases')
    print("Noun phrases are saved as 'custom_noun_phrases.txt' in the script's directory.\n")


def test_custom2(text_name):
    print("\n=============================\n")
    print("Reading the txt file...\n")
    string = open(text_name).read().lower().split("====================\n")
    a = split_sentences2(string)
    print("Running viterbi algorithm... (this may take a long time depending on the test size)\n")
    viterbi_results_a = []
    for sentence in tqdm(a):
        viterbi_results_a.append(viterbi(sentence, distinct_train_tags, WLP, TT, distinct_train_words))

    print("--> POS Tagging completed for", len(viterbi_results_a), "sentences.")
    custom = open(test_path).read().lower().rstrip("\n").split("\n")
    custom_tag_words = split_words_tags(custom)
    custom_set = create_set(custom_tag_words)
    count = 0
    t_count = 0
    for j in range(len(viterbi_results_a) - 1):
        for t in range(len(viterbi_results_a[j])):
            t_count += 1
            x1 = custom_set[j][t]
            x2 = viterbi_results_a[j][t]
            if x1 == x2:
                count += 1
    print("--> with ", (count / t_count) * 100, "%", " accuracy")
    print("Saving the results...")
    write_tags_out(viterbi_results_a, 'custom_tags')
    print("Tagging result is saved as 'custom_tags.txt' in the script's directory.\n")
    print("Extracting the Noun Phrases")
    np_a = noun_phrases(viterbi_results_a)
    print("--> Extraction completed.\n", len(np_a), "noun phrases found.")
    print("Saving the noun phrases...")
    write_np_out(np_a, 'custom_noun_phrases')
    print("Noun phrases are saved as 'custom_noun_phrases.txt' in the script's directory.\n")


def test_custom3(text_name):
    print("\n=============================\n")
    print("Reading the txt file...\n")
    string = open(text_name).read().lower()
    a = split_sentences4(string)
    print("Running viterbi algorithm... (this may take a long time depending on the test size)\n")
    viterbi_results_a = []
    for sentence in tqdm(a):
        viterbi_results_a.append(viterbi(sentence, distinct_train_tags, WLP, TT, distinct_train_words))

    print("--> POS Tagging completed for", len(viterbi_results_a), "sentences.")
    print("Saving the results...")
    write_tags_out(viterbi_results_a, 'custom_tags')
    print("Tagging result is saved as 'custom_tags.txt' in the script's directory.\n")
    print("Extracting the Noun Phrases")
    np_a = noun_phrases(viterbi_results_a)
    print("--> Extraction completed.\n", len(np_a), "noun phrases found.")
    print("Saving the noun phrases...")
    write_np_out(np_a, 'custom_noun_phrases')
    print("Noun phrases are saved as 'custom_noun_phrases.txt' in the script's directory.\n")


def write_tags_out(viterbi_res, file_name):
    filename = file_name + ".txt"
    file = open(filename, "w+")
    for i in range(len(viterbi_res)):
        sentence = viterbi_res[i]
        for j in range(len(sentence)):
            wt = sentence[j]
            line = wt[0] + "/" + wt[1] + "\n"
            file.write(line)


def write_np_out(np_array, file_name):
    filename = file_name + ".txt"
    file = open(filename, "w+")
    for i in range(len(np_array)):
        phrase = np_array[i]
        words = ""
        tags = ""
        for j in range(len(phrase)):
            words += phrase[j][0] + " "
            tags += phrase[j][1] + " - "
        out = "Phrase" + str(i) + ":\n" + words + "\n" + tags + "\n"
        file.write(out)


print("\n=============================\n")
print("Reading the train set...\n")
test_path = "genia-test.txt"
train_path = "genia-train.txt"
train = open(train_path).read().lower().rstrip("\n").split("\n")
test = open(test_path).read().lower().rstrip("\n").split("\n")
train_tag_words = split_words_tags(train)
train_set = create_set(train_tag_words)
test_tag_words = split_words_tags(test)
test_set = create_set(test_tag_words)

train_tagged_words = [tup for sent in train_set for tup in sent]
test_tagged_words = [tup for sent in test_set for tup in sent]

train_string = open(train_path).read().lower().split("====================\n")
train_sentences = split_sentences(train_string)
distinct_train_words = list(set([pair[0] for pair in train_tagged_words]))
distinct_train_tags = list(set([pair[1] for pair in train_tagged_words]))
tags = {tag for word, tag in train_tagged_words}
print("\n=============================\n")
print("Generating Word Likelihood Probabilities Table...\n")
WL = word_likelihood(train_sentences, tags)

WLP = WL.copy()
column_names = WL.columns.values[1:]
for column in tqdm(column_names):
    probs = []
    col_values = WL[column].copy()
    for row in col_values:
        if row == 0:
            probs.append(0)
            continue
        p = row / sum(col_values)
        probs.append(p)

    WLP[column] = probs

print("\n=============================\n")
print("Generating Tag Transition Table... (This may take a while)\n")
tags_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
for i, t1 in enumerate(tqdm(list(tags))):
    for j, t2 in enumerate(list(tags)):
        a = t2_given_t1(t2, t1, train_tagged_words)
        tags_matrix[i, j] = a[0] / a[1]

TT = pd.DataFrame(tags_matrix, columns=list(tags), index=list(tags))


def main():
    inx = -1
    print("Model is trained and ready for action.")
    while inx != 0:
        print("=============================\n")
        print("Enter\n",
              "'1' for tagging the genia-test set.\n",
              "'2' for tagging a custom .txt file.\n",
              "'0' to exit.\n",
              "Waiting for input:")
        inx = int(input())
        if inx == 0:
            break
        elif inx == 1:
            test_genia()
        elif inx == 2:
            print("Choose .txt format\n 1: Regular text.\n 2: Genia format with tags\n 3: Lined format without tags")
            format_input = int(input())
            print("Enter .txt file name (Remember it should be on the script's directory):")
            text_name_in = str(input()) + ".txt"
            if format_input == 1:
                test_custom1(text_name_in)
            elif format_input == 2:
                test_custom2(text_name_in)
            elif format_input == 3:
                test_custom3(text_name_in)
            else:
                print("Error: wrong_input")

            print("press enter to continue")
            input()


main()
