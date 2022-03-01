from math import inf
from sys import argv


def correct_word(my_word: str, right_word: str) -> str:
    """ Получаем из одного слова второе (казалось бы контрпримера нет (а нет есть брух)) """

    my_word = ' ' + my_word + ' '
    right_word = ' ' + right_word + ' '

    # добавляем пустые символы, чтобы не смотреть на индексы
    if len(right_word) < len(my_word):
        right_word = right_word + ' ' * (len(my_word) - len(right_word))
    elif len(my_word) < len(right_word):
        my_word = my_word + ' ' * (len(right_word) - len(my_word))

    for i in range(len(my_word)):
        if my_word[i] != right_word[i]:  # чекаем перестановка, пропущенная буква или лишняя
            if my_word[i + 1] == right_word[i] and my_word[i] == right_word[i + 1]:  # это перестановка
                return my_word[:i] + my_word[i + 1] + my_word[i] + my_word[i + 2:]

            if my_word[i + 1] == right_word[i]:  # вставлена лишняя буква
                return my_word[:i] + my_word[i + 1:]

            if my_word[i] == right_word[i + 1]:  # пропущена буква
                return my_word[:i] + right_word[i] + my_word[i:]

            return my_word[:i] + right_word[i] + my_word[i + 1:]  # неверная буква

    return my_word


def get_corrections(my_word: str, right_word: str):
    """ меняем одно слово на второе и получаем последовательные изменения """

    new_word = correct_word(my_word, right_word)

    if my_word == right_word:
        return []

    corrections = []
    counter = 0
    while new_word != right_word:
        counter += 1
        new_word = correct_word(new_word, right_word).replace(' ', '')
        corrections.append(new_word)

    return corrections


def get_best(word, dictioinary):
    best_word, best_corrections = '', ''
    best_len = inf
    for elem in dictioinary:
        new_corr = get_corrections(word, elem)
        if len(new_corr) < best_len:
            best_word = elem
            best_corrections = new_corr
            best_len = len(new_corr)

    return best_word, best_corrections


def main():
    """ Решение через последовательное получения из одного слова другое (перебор всех пар) """
    dictionary = []
    with open('dict.txt', 'r', encoding='utf-8') as file:
        for _ in range(62027):
            dictionary.append(file.readline().split()[0])

    with open('asd.txt', 'w', encoding='utf-8') as out_file:
        with open(file_name, 'r', encoding='utf-8') as file:
            for _ in range(10000):
                word = file.readline().split()[0]
                dict_word, first_corrections = get_best(word, dictionary)
                dict_word, second_corrections = get_best(word[::-1], dictionary[::-1])

                if len(first_corrections) < len(second_corrections):
                    corrections = first_corrections
                else:
                    corrections = [elem[::-1] for elem in second_corrections]

                init_word = word[:]

                if len(corrections) == 0:
                    out = f'{init_word} 0'
                elif len(corrections) == 1:
                    out = f'{init_word} 1 {dict_word}'
                elif len(corrections) == 2:
                    out = f'{init_word} 2 {corrections[0]} {corrections[1]}'
                else:
                    out = f'{init_word} 3+'

                print(corrections)
                out_file.write(out + '\n')


if __name__ == '__main__':
    file_name = argv[1]
    main(file_name)
