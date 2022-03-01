from math import sqrt, inf
from sys import argv

ALPHABET_SIZE = 32
import numpy as np


class Vector:
    """ Математического вектора """

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


def levenshtein_ratio_and_distance(s, t, ratio_calc=False):
    # https://www.datacamp.com/community/tutorials/fuzzy-string-python
    # потому что впадлу самому в матрицах разбираться брух

    rows = len(s)
    cols = len(t)
    distance = np.zeros((rows + 1, cols + 1), dtype=int)

    for i in range(1, rows):
        for k in range(1, cols):
            distance[i][0] = i
            distance[0][k] = k

    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
            else:
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row - 1][col] + 1, distance[row][col - 1] + 1,
                                     distance[row - 1][col - 1] + cost)

    ratio = ((len(s) + len(t)) - distance[row][col]) / (len(s) + len(t))

    return ratio, distance[row][col]


def read_dictionary(file_name: str):
    """ Читаем весь словарь в виду векторов """
    words = []

    with open(file_name, 'r', encoding='utf-8') as file:
        for _ in range(62027):
            word = file.readline().split()[0]

            vector_word = Vector(word, [])
            words.append(vector_word)

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


def find_nearest_by_levenshtein(request: str, dictionary: list):
    nearest, max_ratio, dist = '', -1, 0
    for word in dictionary:
        new_ratio, new_dist = levenshtein_ratio_and_distance(word.word, request)
        if new_ratio > max_ratio:
            max_ratio = new_ratio
            nearest = word
            dist = new_dist
    return nearest.word, dist, nearest


def main(file_name: str):
    """ Решение через расстояние Левенштейна """
    dictionary = read_dictionary('dict.txt')

    with open(f'{file_name}.out', 'w', encoding='utf-8') as out_file:
        with open(file_name, 'r', encoding='utf-8') as file:
            for _ in range(100000):
                word = file.readline().split()[0]
                ratio, mistakes, nearest = find_nearest_by_levenshtein(word, dictionary)
                out_file.write(f'{word} \t- {nearest.word} \t- {mistakes}\n')


if __name__ == '__main__':
    file_name = argv[1]
    main(file_name)
