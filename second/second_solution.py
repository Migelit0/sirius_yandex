import json
import csv
from math import sqrt
import numpy as np

FORBIDDEN_SYMBOLS = ['.', '"', ',', '(', ')', '\t', '\n', '-', '«', '№', '»']
ALPHABET_SIZE = 740


def format_string(string: str) -> list:
    for elem in FORBIDDEN_SYMBOLS:
        string = string.replace(elem, ' ')

    return string.lower().split()


class Vector:
    def __init__(self, word: str, coords: list, is_word=True, manual_word=None):
        self.coords = [0] * ALPHABET_SIZE

        if is_word:
            self.word = word
            self.update_coords()
        else:
            self.coords = coords
            self.word = manual_word

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
        try:
            return abs(self * other) / (abs(self) * abs(other))
        except Exception:
            return None


def levenshtein_ratio_and_distance(s, t, ratio_calc=False):
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


def find_nearest_by_levenshtein(request: str, dictionary: list):
    nearest, max_ratio, dist = '', -1, 0
    for word in dictionary:
        new_ratio, new_dist = levenshtein_ratio_and_distance(word, request)
        if new_ratio > max_ratio:
            max_ratio = new_ratio
            nearest = word
            dist = new_dist
    return nearest, dist


def find_nearest_by_cos(request: Vector, dictionary: list):
    nearest, max_cos = '', -1
    for word in dictionary:
        new_cos = word ^ request
        if new_cos is None:
            continue

        if new_cos > max_cos:
            max_cos = new_cos
            nearest = word
    return nearest


def precalculate():
    """ Выполняем все необходимые вычисления для итогового подсчета """
    # переводим все названия в векторы и делаем алфавит по словам, которые используеются
    words = set()
    with open('data/universities.txt', 'r', encoding='utf-8') as file:
        for _ in range(747):
            string = file.readline()
            word_list = format_string(string)
            for elem in word_list:
                if len(elem) > 4:
                    words.add(elem)

    with open('data/alphabet.txt', 'w', encoding='utf-8') as file:
        file.write(';;'.join(words))

    ALPHABET_SIZE = len(words)

    alphabet = open('data/alphabet.txt', 'r', encoding='utf-8').read().split(';;')

    # переводим в векторы все слова, где ошибка больше 1 по Левинштейну (изначально просто больше 30 было)

    aboba_words = {}

    with open('data/answers.txt.csv', 'r', encoding='utf-8') as csv_file:
        for _ in range(50000):
            row = csv_file.readline().split(' ;; ')
            if int(row[2]) > 1:
                name_list = format_string(row[0])
                coords = [0] * ALPHABET_SIZE
                for word in name_list:
                    nearest_word, ratio = find_nearest_by_levenshtein(word, alphabet)
                    coords[alphabet.index(nearest_word)] += 1
                aboba_words[row[0]] = coords

    with open('data/aboba_words.json', 'w', encoding='utf-8') as json_file:
        json.dump(aboba_words, json_file, ensure_ascii=False)

    vectorized_dictionary = {}

    # переводим в векторы все названия из словаря
    all_universities = open('data/universities.txt', 'r', encoding='utf-8').read().split('\n')

    for name in all_universities:
        name_list = format_string(name)

        coords = [0] * ALPHABET_SIZE
        for word in name_list:
            if word in alphabet:
                coords[alphabet.index(word)] += 1
            else:
                print(word)
        vectorized_dictionary[name] = coords

    with open('data/vectorizer_univers.json', 'w', encoding='utf-8') as json_file:
        json.dump(vectorized_dictionary, json_file, ensure_ascii=False)


def main(is_calculated=False):
    # если мы все заранее посчитали, то просто для каждого ошибочного ищем самое близкое верное
    alphabet = open('data/alphabet.txt', 'r', encoding='utf-8').read().split(';;')
    ALPHABET_SIZE = len(alphabet)

    with open('data/vectorizer_univers.json', 'r', encoding='utf-8') as json_file:
        vectorized_dictionary = json.load(json_file)

    vectorized_dictionary = [Vector('', vectorized_dictionary[i], is_word=False, manual_word=i) for i in
                             vectorized_dictionary]
    print(len(vectorized_dictionary))

    with open('data/aboba_words.json', 'r', encoding='utf-8') as json_file:
        aboba_words = json.load(json_file)

    aboba_words = [Vector('', aboba_words[i], is_word=False, manual_word=i) for i in aboba_words]
    print(len(aboba_words))

    corrected_abobas = {}
    for word in aboba_words:
        best_word = find_nearest_by_cos(word, vectorized_dictionary)
        corrected_abobas[word] = best_word

    all_data = open('data/answers.txt.csv', 'r', encoding='utf-8').read().split('\n')
    new_data = []

    for elem in all_data:
        row = elem.split(' ;; ')
        if len(row) != 3:
            continue
        if int(row[2]) > 1:
            if row[0] in [i.word for i in corrected_abobas]:
                for a in corrected_abobas:
                    if a.word == row[0]:
                        row[1] = corrected_abobas[a].word
        new_data.append(' ;; '.join(row))

    with open('data/answers02.csv', 'w', encoding='utf-8') as file:
        file.write('\n'.join(new_data))


if __name__ == '__main__':
    # все уже посчитано заранее и лежит в data/ , поэтому считать заранее не нужно

    # Кратко суть в ридми основном
    main(is_calculated=True)
