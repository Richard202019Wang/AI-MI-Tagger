# AI-MI-Tagger

Here's a detailed breakdown of what the project does:

Reading Training Files: The function read_given_file() reads in the provided training files. Each file presumably contains a sequence of words along with their corresponding POS tags, which are separated by a colon. The end of a sentence is recognized based on specific punctuation marks like '"', "!", ".", "?".

Counting Frequency Data: The function count_separate_numbers() processes the loaded sentences and counts the frequency of different events, such as the overall tag count, tag-word pairs, first tags in sentences, and the last tag seen prior to each tag.

Probability Calculation: The get_all_probabilities() function uses the counted frequency data to calculate the Initial, Transition, and Observation probabilities needed for the Viterbi algorithm.

Implementing the Viterbi Algorithm: The functions initial_probabilities(), Transition_matrix(), Observation_matrix(), update_new_prev_prob(), and Viterbi() implement the Viterbi algorithm itself. These functions calculate the most likely sequence of POS tags for a given sentence based on the previously calculated probabilities.

Result Generation and Output: The function get_result() uses the results of the Viterbi algorithm to generate the most probable POS tag sequence for a given input sentence. This sequence is then written to the output file by the write_output() function.

Processing Test Data: The script reads in test data and applies the above steps to each sentence in the test data, writing the results to an output file.

So, in summary, this project is about training a Hidden Markov Model with the Viterbi algorithm for POS tagging, using a given set of training data. It then applies the trained model to a test dataset and outputs the predicted POS tags.
