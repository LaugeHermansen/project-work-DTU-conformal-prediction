from Bio import pairwise2 as pw2
from textwrap import fill, shorten, wrap
import os

def compare(filename_old, filename_new, path = 'latex_comparisons/', print_comparison = True, width = 40, create_new_comparison_files = False):
    """
    function that takes in the filenames of
    the old and new version of the section and compares them

    Parameters
    ---------
     - filename_old: the filename of the old version (omit the '.tex'), e.g. '4_methods'
     - filename_new: the filename of the new version (omit the '.tex'), e.g. 'methods'
     - path: the path to the root of this part of the project - include "/"
     - print_comparison: you guessed it
     - width: the column width
     - create_new_comparison_files: requires sequence alignment of the files again - takes time
    
    Output
    ----------
     - a looong string that compares the two versions of the section
      >   - actually two strings - one that is uncolored and one that is colored

    """

    raw_output = []
    colored_output = []

    def output(string):
        raw_output.append(string)
        colored_output.append(string)

    if 'compare_data_files' not in os.listdir():
        os.mkdir('compare_data_files')

    f_new = f'{path}compare_data_files/{filename_old}_vs_{filename_new}-new.txt'
    f_old = f'{path}compare_data_files/{filename_old}_vs_{filename_new}-old.txt'
    
    try:
        assert not create_new_comparison_files
        with open(f_new) as file:
            al_new = file.read()
        with open(f_old) as file:
            al_old = file.read()

    except:

        with open(f'{path}02466-project-work-conformal-prediction/chapters/{filename_new}.tex') as file:
            methods_new = file.read()
        with open(f'{path}Project_work_Conformal_prediction/sources/{filename_old}.tex') as file:
            methods_old = file.read()

        a = pw2.align.globalxx(methods_new, methods_old, one_alignment_only = True, )[0]
        al_new, al_old = a[:2]

        with open(f_new, 'w') as file:
            file.write(al_new)
        with open(f_old, 'w') as file:
            file.write(al_old)

    dif = [[-1,"\n"]]

    def colored(r, g, b, text):
        return f"\033[38;2;{r};{g};{b}m{text}\033[38;2;255;255;255m"

    def text_wrap_help(text, i, width):
        n = len(text)
        if i < n: return text[i] + " "*(width - len(text[i]))
        else:
            return " "*width

    def print_cum_sum(cum_sum, width = 70):
        cum_sum[0] = cum_sum[0].strip()
        cum_sum[1] = cum_sum[1].strip()
        if cum_sum != ["", ""]:
            raw_output.append('-'*(width*2+3))
            colored_output.append(' '*(width*2+3))
            text1 = wrap(cum_sum[0], width = width)
            text2 = wrap(cum_sum[1], width = width)
            for i in range(max(len(text1),len(text2))):
                t1 = text_wrap_help(text1,i,width)
                t2 = text_wrap_help(text2,i,width)
                if t1 != t2: output(f'{t1} | {t2}')
                else: output(t1)
            raw_output.append('-'*(width*2+3))
            colored_output.append(' '*(width*2+3))

    def print_0_text(text, width):

        for t in text.split('\n'):
            if len(t) > width:
                out = shorten(t, width = width//2 + 7) + shorten(t[::-1], width = width//2)[-6::-1]
            else:
                out = t
            raw_output.append(out)
            colored_output.append(colored(120,120,120,out))

    def crappy_helper_function(id, ap):
        if dif[-1][0] == id:
            dif[-1][1] += ap
        else:
            dif.append([id,ap])

    for i, (m1,m2) in enumerate(zip(al_new, al_old)):
        if m1 == m2:    crappy_helper_function(0, m1)
        elif m1 == '-': crappy_helper_function(1, m2)
        elif m2 == '-': crappy_helper_function(2, m1)
        else:           crappy_helper_function(3, f'({m1}|{m2})')

    cum_sum = ["", ""]

    for [id, text] in dif[1:]:
        if id == 0 and len(text) >=width/2:
            print_cum_sum(cum_sum, width)
            cum_sum = ["", ""]
            print_0_text(text, width = width*2)

        elif id == 0:
            cum_sum[0] += text
            cum_sum[1] += text
        elif id == 1: cum_sum[0] += text
        elif id == 2: cum_sum[1] += text
        else:
            raise ValueError('hovsa')

    print_cum_sum(cum_sum, width = width)

    raw_output = '\n'.join(raw_output)
    colored_output = '\n'.join(colored_output)


    if print_comparison: print(colored_output)

    return raw_output, colored_output

if __name__ == "__main__":
    compare('4_methods', 'methods', print_comparison=True, create_new_comparison_files=False)


