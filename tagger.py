import argparse
import numpy as np


def read_given_file(training_file_list):
    """
    Get all whole sentences of every training file which end with '"', "!", ".", "?"
    :param training_file_list:
    :return:
    """
    # Initialize an empty list to store all the lines
    total_lines = []
    # Loop through each file in the training_file_list
    for file in training_file_list:
        with open(file, 'r') as f:
            data = f.readlines()
        # Initialize an empty list to store the current sentence
        whole_sentence = []
        for l in data:
            acc_l = l.strip("\n")
            # Check if the sentence end with punctuation
            if acc_l.split(" : ")[0] not in {'"', "!", ".", "?"}:
                whole_sentence += [acc_l]
            else:
                whole_sentence += [acc_l]
                total_lines.append(whole_sentence)
                whole_sentence = []
        # If there is an unfinished sentence, add it to total_lines
        if whole_sentence:
            total_lines.append(whole_sentence)

    return total_lines


def count_separate_numbers(total_lines):
    total_tag, total_words, total_start, total_last_tag, total_hidden = {}, {}, {}, {}, []

    for l in total_lines:
        last = None
        # Loop through each word-tag pair in the sentence
        for group_num, line in enumerate(l):
            # Split the line into word and tag
            acc_word, acc_tag = line.split(" : ")
            # If it's not the first word in the sentence, update the total_last_tag dictionary
            if group_num != 0:
                total_last_tag.setdefault(acc_tag, {})
                total_last_tag[acc_tag][last] = total_last_tag[acc_tag].get(last, 0) + 1
            else:
                # If it's the first word in the sentence, update the total_start dictionary
                total_start[acc_tag] = total_start.get(acc_tag, 0) + 1

            # Add the unseen tag in the total_hidden
            if acc_tag not in total_hidden:
                total_hidden.append(acc_tag)

            # Update the total_tag and total_words dictionaries
            last = acc_tag
            total_tag[acc_tag] = total_tag.get(acc_tag, 0) + 1
            total_words.setdefault(acc_tag, {})
            total_words[acc_tag][acc_word] = total_words[acc_tag].get(acc_word, 0) + 1

    return total_tag, total_words, total_start, total_last_tag, total_hidden


def get_all_probabilities(total_tag, total_words, total_start, total_last_tag, total_sentences):
    total_words_num = sum([len(words) for words in total_words.values()])
    Observation_prob = {}
    smooth_param_obs = 0.025

    for tags in total_words:
        Observation_prob[tags] = {}
        for words in total_words[tags]:
            # Calculate the observation probability using smoothing(a way to increase the accuracy I google online) and store it in the Observation_prob dictionary
            Observation_prob[tags][words] = (total_words[tags][words] + smooth_param_obs) / (total_tag[tags] + total_words_num * smooth_param_obs)

    Transition_prob = {}
    for tags in total_last_tag:
        Transition_prob[tags] = {}
        for words in total_last_tag[tags]:
            # Calculate the transition probability and store it in the Transition_prob dictionary
            Transition_prob[tags][words] = total_last_tag[tags][words] / total_tag[words]

    Initial_prob = {}
    for tags in total_start:

        # Calculate the initial probability and store it in the Initial_prob dictionary
        Initial_prob[tags] = total_start[tags] / total_sentences

    return Observation_prob, Transition_prob, Initial_prob


######## Algorithm part #######

def initial_probabilities(Observation_prob, Initial_prob, total_lines, total_hidden):
    """
    Determine values for Step 0
    :param Observation_prob:
    :param Transition_prob:
    :param Initial_prob:
    :param total_lines:
    :param total_hidden:
    :return:
    """
    # Initialize probability and previous state matrices
    acc_prob, acc_prev = np.zeros((len(total_lines), len(total_hidden))), np.zeros((len(total_lines), len(total_hidden)))
    check_start = 0
    # Check if the first word of total_lines is in the Observation_prob and Initial_prob
    for tag in total_hidden:
        if total_lines[0] in Observation_prob[tag] and tag in Initial_prob:
            check_start += 1
            break

    # Calculate the initial probabilities for each hidden state
    for idx in range(len(total_hidden)):
        if check_start == 1 and total_lines[0] in Observation_prob[total_hidden[idx]] and total_hidden[idx] in Initial_prob:
            acc_prob[0, idx] = Initial_prob[total_hidden[idx]] * Observation_prob[total_hidden[idx]][total_lines[0]]
        elif check_start == 0 and total_hidden[idx] in Initial_prob:
            acc_prob[0, idx] = Initial_prob[total_hidden[idx]]
        acc_prev[0, idx] = None

    return acc_prev, acc_prob


def Transition_matrix(Transition_prob, total_hidden, total_tag):
    transition_matrix = []
    smooth_param_trans = 0.01

    for i in range(len(total_hidden)):
        acc_transition = []
        for j in range(len(total_hidden)):
            if not (total_hidden[i] in Transition_prob and total_hidden[j] in Transition_prob[total_hidden[i]]):
                # acc tag pair not found in transition probability, I use a new smooth transition probability
                acc_transition.append((1 + smooth_param_trans) / (total_tag[total_hidden[j]] + smooth_param_trans * len(total_hidden)))

            else:
                acc_transition.append(Transition_prob[total_hidden[i]][total_hidden[j]])

        transition_matrix.append(acc_transition)
    return np.array(transition_matrix)


def Observation_matrix(Observation_prob, total_hidden, observe_flag, acc_word):
    observation_matrix = []
    for i in range(len(total_hidden)):
        if not observe_flag: # If current word is not in training data
            observation_matrix.append(1) # I want to ignore the observation probability and let the transition
            # probabilities and previous probabilities help determine the most likely tag.
        elif acc_word not in Observation_prob[total_hidden[i]]:
            observation_matrix.append(0) # This is to indicate that the probability of the word being associated with
            # that tag is extremely low or nonexistent based on the training data.

        else:
            observation_matrix.append(Observation_prob[total_hidden[i]][acc_word])

    return np.array(observation_matrix)[:, np.newaxis] # Convert the matrix to array


def update_new_prev_prob(acc_prob, acc_prev, Observation_prob, Transition_prob, steps, total_hidden, total_tag, total_lines):
    observe_flag = 0
    for tag in total_hidden:
        if tag in Transition_prob:
            if total_lines[steps] in Observation_prob[tag]:
                observe_flag += 1
                break

    # Create the needed matrixes as the pesudocode in lecture by calling my helpers

    observation_matrix = Observation_matrix(Observation_prob, total_hidden, observe_flag, total_lines[steps])
    transition_matrix = Transition_matrix(Transition_prob, total_hidden, total_tag)
    acc_prob_matrix = np.tile(acc_prob[steps - 1], (len(total_hidden), 1))

    # Calculate argmax_x

    argmax_x_matrix = np.multiply(np.multiply(acc_prob_matrix, transition_matrix), observation_matrix)

    # Find each tag's maximum probability with normalization
    max_prob = np.amax(argmax_x_matrix, axis=1)
    max_prob /= np.sum(max_prob)

    # Update acc_prob, acc_prev
    acc_prev[steps], acc_prob[steps] = np.argmax(argmax_x_matrix, axis=1), max_prob

    return acc_prev, acc_prob


def Viterbi(Observation_prob, Transition_prob, Initial_prob, total_hidden, total_tag, total_lines):
    """
    For steps 1 to length(total_lines)-1
    :param Observation_prob:
    :param Transition_prob:
    :param Initial_prob:
    :param total_hidden:
    :param total_tag:
    :param total_lines:
    :return:
    """
    # Initialize probabilities and previous states for step 0
    acc_prev, acc_prob = initial_probabilities(Observation_prob, Initial_prob, total_lines, total_hidden)

    # Update probabilities and previous states for steps 1 to len(total_lines) - 1
    for i in range(1, len(total_lines)):
        acc_prev, acc_prob = update_new_prev_prob(acc_prob, acc_prev, Observation_prob, Transition_prob, i,
                                                  total_hidden, total_tag, total_lines)

    return acc_prev, acc_prob


def get_result(total_lines, pre_prob, acc_prob, total_hidden):
    # Find the index of the highest probability in the last step
    highest_prob_idx = np.argmax(acc_prob[-1])
    result = [total_lines[-1] + " : " + total_hidden[highest_prob_idx]]

    # Backtrack through the previous states to get the most probable tags for each word
    for i in range(len(total_lines) - 2, -1, -1):
        highest_prob_idx = int(pre_prob[i + 1, highest_prob_idx])
        result.append(total_lines[i] + " : " + total_hidden[highest_prob_idx])

    return result[::-1]


def write_output(training_file_list, test_file, result_file):
    # Read training data and count separate numbers
    read_data = read_given_file(training_file_list)
    total_tag, total_words, total_start, total_last_tag, total_hidden = count_separate_numbers(read_data)
    # Calculate probabilities based on the training data
    Observation_prob, Transition_prob, Initial_prob = get_all_probabilities(total_tag, total_words, total_start, total_last_tag, len(read_data))
    total_test_line = []
    whole_sentence = []

    # Read test data
    with open(test_file) as f:
        data = f.readlines()

    # Split test data into sentences
    for l in data:
        acc_l = l.strip("\n")
        if acc_l.split(" : ")[0] not in {'"', "!", ".", "?"}:
            whole_sentence += [acc_l]
        else:
            whole_sentence += [acc_l]
            total_test_line.append(whole_sentence)
            whole_sentence = []
    if whole_sentence:
        total_test_line.append(whole_sentence)

    # Write the results to the output file
    with open(result_file, "w") as f:
        for test_lines in total_test_line:
            pre_prob, acc_prob = Viterbi(Observation_prob, Transition_prob, Initial_prob, total_hidden, total_tag, test_lines)
            write_lines = get_result(test_lines, pre_prob, acc_prob, total_hidden)
            for lines in write_lines:
                f.write(lines + "\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainingfiles",
        action="append",
        nargs="+",
        required=True,
        help="The training files."
    )
    parser.add_argument(
        "--testfile",
        type=str,
        required=True,
        help="One test file."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file."
    )
    args = parser.parse_args()

    training_list = args.trainingfiles[0]
    print("training files are {}".format(training_list))

    print("test file is {}".format(args.testfile))

    print("output file is {}".format(args.outputfile))


    print("Starting the tagging process.")

    write_output(training_list, args.testfile, args.outputfile)
