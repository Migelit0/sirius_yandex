from math import sqrt, inf
from sys import argv

ALPHABET_SIZE = 32
import numpy as np


def levenshtein_ratio_and_distance(s, t, ratio_calc=False): # https://www.datacamp.com/community/tutorials/fuzzy-string-python
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s) + 1
    cols = len(t) + 1
    distance = np.zeros((rows, cols), dtype=int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1, cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
                cost = 0  # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row - 1][col] + 1,  # Cost of deletions
                                     distance[row][col - 1] + 1,  # Cost of insertions
                                     distance[row - 1][col - 1] + cost)  # Cost of substitutions

    Ratio = ((len(s) + len(t)) - distance[row][col]) / (len(s) + len(t))

    return Ratio, distance[row][col]


class Vector:
    def __init__(self, word: str, coords: list, is_word=True):
        self.coords = [0] * ALPHABET_SIZE

        if is_word:
            self.word = word
            self.update_coords()
        else:
            self.coords = coords

    def update_coords(self):
        for letter in self.word:
            try:
                self.coords[ord(letter) - 1072] += 1
            except Exception:
                pass

    def __abs__(self):
        """ Модуль вектора """
        return sqrt(sum([x ** 2 for x in self.coords]))

    def __mul__(self, other):
        """ Перемножение двух векторов """
        new_coords = [0] * ALPHABET_SIZE

        for i in range(ALPHABET_SIZE):
            new_coords[i] = self.coords[i] * other.coords[i]

        return sum(new_coords)

    def __xor__(self, other):
        """ Находим косинус между двумя векторами """
        return abs(self * other) / (abs(self) * abs(other))


def read_dictionary(file_name: str):
    words = {}

    with open(file_name, 'r', encoding='utf-8') as file:
        for _ in range(757):
            word = ' '.join(file.readline().split()).replace('\n', '')

            vector_word = Vector(word, [])
            words[vector_word] = word.lower()

    return words


def find_nearest(request: str, dictionary: list):
    request_vector = Vector(request, [])

    nearest, max_cos = '', -1
    for word in dictionary:
        new_cos = word ^ request_vector
        if new_cos > max_cos:
            max_cos = new_cos
            nearest = word
    return nearest.word


def find_nearest_by_levenshtein(request: str, dictionary: dict):
    nearest, max_ratio, dist = '', -1, 0
    for word in dictionary:
        new_ratio, new_dist = levenshtein_ratio_and_distance(word.word, request)
        if new_ratio > max_ratio:
            max_ratio = new_ratio
            nearest = word
            dist = new_dist
    return nearest.word, dist, nearest


def main(file_name: str):
    dictionary = read_dictionary('universities.txt')

    with open(f'{file_name}.out', 'w', encoding='utf-8') as out_file:
        with open(file_name, 'r', encoding='utf-8') as file:
            for _ in range(10000):
                word = ' '.join(file.readline().split()).replace('\n', '')
                ratio, mistakes, nearest = find_nearest_by_levenshtein(word.lower(), dictionary)
                print(f'{word} \t- {nearest.word} \t- {mistakes}')
                out_file.write(f'{word} \t- {nearest.word} \t- {mistakes}\n')


if __name__ == '__main__':
    # Просто поиск по Левинштейну
    # Хранится все в векторах и делал спеша, поэтому не успел на просто нормальные строки переделать, виноват
    file_name = argv[1]
    main(file_name)
