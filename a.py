import numpy as np
import math
import statistics as st
from collections import Counter
import pdb


def load_from_csv(inFile):
    """Used to load a CSV file from disk.
    The returned value is a list of lists"""

    my_array = np.loadtxt(inFile, delimiter=",")
    return(my_array)


def get_distance(list_one, list_two):
    """Returns the Euclidean distance between the two arguments
    Both the arguments should be of type list
    Return type: float"""

    nums = np.subtract(list_one, list_two)   # elementwise subtraction
    square_nums = list(map(lambda x: x ** 2, nums))   # square resultant list
    sum_list = sum(square_nums)  # elementwise addition of resultant list
    # limiting precision to 3 decimal places with.3f - Change if required
    euclid_dist = float("{0:.3f}".format(math.sqrt(sum_list)))
    return euclid_dist


def get_standard_deviation(matrix, col_num):
    """Returns the standard deviation (number) of the values in a given column
    of the matrix
    Return type: float"""

    all_col_avg = np.average(matrix, 0)  # 0 = col average; 1 = row average
    col_avg = all_col_avg[col_num]
    new_matrix = matrix[:, col_num] - col_avg
    squared = list(map(lambda x: x ** 2, new_matrix))
    sum_squared = sum(squared)
    return math.sqrt((1 / (len(new_matrix) - 1)) * sum_squared)


def get_standardised_matrix(matrix):
    """Returns a standardised matrix based on the application of standard  
    deviation of each of the columns to the matrix
    Return type: matrix (list of lists)"""

    all_col_avg = np.average(matrix, 0)
    rows, cols = matrix.shape
    std_dev_all_col = []
    # creating an zero matrix with same shape as input matrix
    std_matrix = np.zeros([rows, cols])
    for col in range(cols):
        std_dev_all_col.append(get_standard_deviation(matrix, col))
        for row in range(rows):
            std_matrix[row, col] = (matrix[row, col] - all_col_avg[col]) / std_dev_all_col[col]
    return std_matrix


def get_k_nearest_labels(input_list, learning_data, learning_data_labels, k):
    """Returns the k nearest data labels from learning_data_labels for the given
    input_list against the learning_data.
    Return type: list.
    Implementation described in comments"""

    # For each row in the data_matrix, pass the row and input_list to
    # "get_distance" function. Append the results returned by
    # "get_distance" function to a "distances" list.

    distances = []
    for data_matrix_row in learning_data:
        distances.append(get_distance(input_list, data_matrix_row))
    # pdb.set_trace()
    # Convert the data_label_matrix to a list. Map the "distances" list
    # with the data_label_matrix to form a dictionary. Keys will be the
    # distance and value will be the data_label value. Sort the dictionary
    # in ascending order to get the first k keys in order. Return the
    # values of the first k keys.

    dist_label_dict = dict(zip(distances, learning_data_labels))
    dist_label_dict_sorted = sorted(dist_label_dict.items(), key = lambda kv:(kv[1], kv[0]))

    # slice the first k items from the sorted dictionary (which becomes a list after sorting)

    k_sorted_items = dist_label_dict_sorted[0:k]
    k_sorted_items_list = []
    for _ in range(k):
        k_sorted_items_list.append(k_sorted_items[_][1])
    return k_sorted_items_list


def get_mode(inList):
    """Return the most frequently occuring item in the list argument"""
    # count_items = dict(Counter(inList))
    # mode_result = sorted(count_items.items(), key=lambda kv: (kv[1], kv[0]))
    # print(mode_result)
    mode_result = st.mode(inList)
    return mode_result


def classify(matrix, learning_data, learning_data_labels, k):
    std_matrix = get_standardised_matrix(matrix)
    std_learning_data = get_standardised_matrix(learning_data)
    data_labels = []
    count = 0
    for std_row in std_matrix:
        # pdb.set_trace()
        data_labels.append(get_mode(get_k_nearest_labels(std_row, std_learning_data, learning_data_labels, k)))
        count += 1
    # print(data_labels)
    # print(count)
    return data_labels
    # classify_mode = get_mode(classify_k_near_labels)
    # return classify_k_near_labels


def get_accuracy(data_labels, correct_data_labels):
    count = 0
    for _ in range(len(data_labels)):
        if data_labels[_] == correct_data_labels[_]:
            count += 1
        else:
            count += 0
    # accuracy = "{0:.3f}".format((count / len(data_labels)) * 100)
    accuracy = (count / len(data_labels)) * 100
    return accuracy


def run_test():
    """Call all the functions to test and verify the expected results"""
    matrix = load_from_csv("Data.csv")
    learning_data = load_from_csv("Learning_Data.csv")
    learning_data_labels = load_from_csv("Learning_Data_Labels.csv")
    correct_data_labels = load_from_csv("Correct_Data_Labels.csv")

    # input_list = [5, 3, 2, 8, 5, 10, 8, 1, 2]
    # k = 3
    # k_nearest_labels = get_k_nearest_labels(input_list, learning_data, learning_data_labels, k)
    # get_mode_value = get_mode(k_nearest_labels)

    for k in range(3,16):
        data_labels = classify(matrix, learning_data, learning_data_labels, k)
        accuracy = get_accuracy(data_labels, correct_data_labels)
        print(f"K={k}, Accuracy={accuracy}",)
    return matrix, learning_data, learning_data_labels, accuracy


# matrix, learning_data, learning_data_labels, accuracy = run_test()

run_test()
