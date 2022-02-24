from sys import argv
from second_solution import get_corrections


def main(input_file_name: str, output_file_name: str):
    """ форматируем файл в формат необходимый для ответа """
    with open(output_file_name, 'w', encoding='utf-8') as out_file:
        with open(input_file_name, 'r', encoding='utf-8') as file:
            for _ in range(10000):
                words = file.readline().split('-')
                print(words)
                init_word = words[0].split()[0]
                dict_word = words[1].split()[0]

                corrections = get_corrections(init_word, dict_word)

                if len(corrections) == 0:
                    out = f'{init_word} 0'
                elif len(corrections) == 1:
                    out = f'{init_word} 1 {dict_word}'
                elif len(corrections) == 2:
                    out = f'{init_word} 2 {corrections[0]} {corrections[1]}'
                else:
                    out = f'{init_word} 3+'

                out_file.write(out + '\n')


if __name__ == '__main__':
    name = argv[1]
    main(f'input_files/{name}.out', f'output_files/{name}_test.out')
