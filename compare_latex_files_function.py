from Bio import pairwise2 as pw2
from textwrap import fill, shorten, wrap
import os

def compare(filename_old, filename_new, path = 'latex_comparisons/', width = 40, save_comparison = False, batch_size = 1500, save_align_files = False):
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
    #helper functions

    def colored(r, g, b, text):
        return f"\033[38;2;{r};{g};{b}m{text}\033[38;2;255;255;255m"

    def text_wrap_help(text, i, width):
        n = len(text)
        if i < n: return text[i] + " "*(width - len(text[i]))
        else:
            return " "*width

    #output variables
    raw_output = []
    colored_output = []

    # helper function - like print
    def output(string):
        raw_output.append(string)
        colored_output.append(string)

    #open files

    f_new = f'{path}compare_data_files/{filename_old}_vs_{filename_new}-new.txt'
    f_old = f'{path}compare_data_files/{filename_old}_vs_{filename_new}-old.txt'

    with open(f'{path}02466-project-work-conformal-prediction/chapters/{filename_new}.tex') as file:
        methods_new = file.read()
    with open(f'{path}Project_work_Conformal_prediction/sources/{filename_old}.tex') as file:
        methods_old = file.read()


    # align the texts ---------------------------------------------------

    i_old = i_new = 0

    al_new = []
    al_old = []

    n_warnings = 0

    counter = 0

    while True:
        counter += 1
        if i_old < len(methods_old) and i_new < len(methods_new):
            a = pw2.align.globalxx(methods_new[i_new:i_new+batch_size], methods_old[i_old:i_old+batch_size], one_alignment_only = True)[0]
            j_old = j_new = 0
            j_old_temp = j_new_temp = 0
            n_following_matches = 0
            al_index = 0
            for al_index_temp, (m_new,m_old) in enumerate(zip(*a[:2])):
                if m_old == m_new and methods_old[i_old + j_old_temp] == methods_new[i_new + j_new_temp]:
                    assert methods_new[i_new + j_new_temp] == m_new
                    assert methods_old[i_old + j_old_temp] == m_old
                    j_old_temp += 1
                    j_new_temp += 1
                    n_following_matches += 1
                    if n_following_matches >= width:
                        j_new = j_new_temp
                        j_old = j_old_temp
                        al_index = al_index_temp
                elif m_old == "-" and (i_old + j_old_temp >= len(methods_old) or methods_old[i_old + j_old_temp] != "-"):
                    j_new_temp += 1
                    n_following_matches = 0
                elif m_new == "-" and (i_new + j_new_temp >= len(methods_new) or methods_new[i_new + j_new_temp] != "-"):
                    j_old_temp += 1
                    n_following_matches = 0
                else:
                    j_old_temp += 1
                    j_new_temp += 1
                    n_following_matches = 0

            if j_old == 0 and j_new == 0:
                j_old = j_new = batch_size
                al_old.append("¤")
                al_new.append("¤")
                n_warnings += 1

            i_old += j_old
            i_new += j_new

            al_new.append(a[0][:al_index+1])
            al_old.append(a[1][:al_index+1])

        elif i_old >= len(methods_old) and i_new < len(methods_new):
            extra_size = len(methods_new[i_new:])
            al_old.append("-"*extra_size)
            al_new.append(methods_new[i_new:])
            break

        elif i_new >= len(methods_new) and i_old < len(methods_old):
            extra_size = len(methods_old[i_old:])
            al_new.append("-"*extra_size)
            al_old.append(methods_old[i_old:])
            break

        else: break

        print("\r", counter, end = "")


    al_new = "".join(al_new)
    al_old = "".join(al_old)


    if save_align_files:
        with open(f_new, 'w') as file:
            file.write(al_new)
        with open(f_old, 'w') as file:
            file.write(al_old)
    
    # start printing --------------------------------------------------------------

    # print the header

    output('\033[38;2;255;255;255m"')
    output(f' --- Comparing "{filename_old}.tex" and "{filename_new}.tex" --- ')
    output('')
    output(f'There were in total {n_warnings} {"warnings" if n_warnings != 1 else "warnings"}')


    # build dif
    #   -  dif is another representation of the alignments that is easier to use for printing
    dif = [[-1,"\n"]]

    def print_cum_sum(cum_sum, width = 70):
        cum_sum[0] = cum_sum[0].strip()
        cum_sum[1] = cum_sum[1].strip()
        if cum_sum != ["", ""]:
            raw_output.append('-'*(width*2+3))
            colored_output.append(' '*(width*2+3))
            text1 = [w for c in cum_sum[0].split('\n') for w in wrap(c, width = width)]
            text2 = [w for c in cum_sum[1].split('\n') for w in wrap(c, width = width)]
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

    assert len(al_old) == len(al_new)
    

    warning = ">>> WARNING: Comparisons may have failed <<<"
    pad = max((width*2-len(warning))//2,0)

    warning = colored(226,39,90," "*pad + warning + " "*pad)

    for i, (m1,m2) in enumerate(zip(al_new, al_old)):
        if m1 == "¤":
            assert m2 == "¤"
            dif.append([4,warning])
        elif m2 == "¤":
            assert m1 == "¤"
            dif.append([4,warning])
        
        elif m1 == m2:  crappy_helper_function(0, m1)
        elif m1 == '-': crappy_helper_function(1, m2)
        elif m2 == '-': crappy_helper_function(2, m1)
        else:           crappy_helper_function(3, f'({m1}|{m2})')

    # dif is created
    # now finally create the output
    # cum_sum should be understood as a cumulative sum of differences

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
        elif id == 4: 
            output(warning)
        else:
            raise ValueError('hovsa')

    print_cum_sum(cum_sum, width = width)


    raw_output = '\n'.join(raw_output)
    # colored_output = '\n'.join(colored_output)

    # done with creating the output

    for text in colored_output:
        print(text)

    return raw_output, '\n'.join(colored_output)

if __name__ == "__main__":
    compare('2_theory', 'theory', batch_size=1500)