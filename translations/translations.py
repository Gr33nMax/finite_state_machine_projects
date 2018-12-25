# Author: Sidorenko Maxim
# VK: vk.com/maksim2009rus
# 1) Generating a Non-Deterministic Finite Automaton by a regular expression.
# 2) Translation of NDFA to DFA.
#
import graphviz as gv


# Standard character classes
D = frozenset("0123456789")
W = D | frozenset("_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
S = frozenset(" \t\n\v\f\r")
SYMBOLS = frozenset("`~!@#%&=;:'\",<>/")
SPECIALS = frozenset("\|.()[]{}*+?-$^")
ALPHABET = W | S | SYMBOLS
ESCAPES = SPECIALS | frozenset("wrtsdfvbn")


def states_generator(index: int) -> str:
    """ Generating unique states. """
    while True:
        index += 1
        state = f"S{index}"
        yield state


def f1(string: str, A: str, B: str, matrix: dict) -> None:
    string += '|'
    counter_round_brackets = 0
    counter_square_brackets = 0
    ind, pos = 0, 0
    substrings = list()
    while ind < len(string):
        if string[ind:ind + 2] in ("\|", "\(", "\)", "\[", "\]"):
            ind += 1
        elif string[ind] == '(':
            if not counter_square_brackets:
                counter_round_brackets += 1
        elif string[ind] == ')':
            if not counter_square_brackets:
                counter_round_brackets -= 1
        elif string[ind] == '[':
            if counter_square_brackets < 1:
                counter_square_brackets += 1
        elif string[ind] == ']':
            counter_square_brackets -= 1
        elif string[ind] == '|':
            if not counter_square_brackets and not counter_round_brackets \
                    and string[pos:ind + 1]:
                substrings.append(string[pos:ind])
                pos = ind + 1
        ind += 1

    for sub in substrings:
        f2(sub, A, B, matrix)


def f2(substr: str, A: str, B: str, matrix: dict) -> None:
    """ Splitting a string into its component parts. """
    substr2 = ""
    substr3 = ""
    round_brackets = 0
    square_brackets = 0
    idx = 0
    while idx < len(substr):
        if len(substr) >= 3 and substr[0] == '\\' and \
                substr[1] in SPECIALS and substr[2] in ('*', '+'):  # "\)+"
            substr2 = substr[:3]
            substr3 = substr[3:]
            break
        elif len(substr) >= 2 and \
                ((substr[0] == '\\' and substr[1] in SPECIALS) or  # "\("
                 (substr[0] in ALPHABET and substr[1] in ('*', '+'))):  # "a+"
            substr2 = substr[:2]
            substr3 = substr[2:]
            break
        elif len(substr) >= 2 and substr[0] in ALPHABET and \
                substr[1] not in ('+', '*'):
            substr2 = substr[0]  # 'a'
            substr3 = substr[1:]
            break
        elif len(substr) == 1 and substr not in SPECIALS:  # 'a'
            substr2 = substr
            break
        elif substr[idx] == '(':
            if not square_brackets:
                round_brackets += 1
        elif substr[idx] == ')' and not substr[idx - 1] == '\\':
            if not square_brackets:
                round_brackets -= 1
            if not round_brackets and not square_brackets:
                if idx < len(substr) - 1:
                    if substr[idx + 1] not in ('*', '+'):  # "(abc)"
                        substr2 = substr[:idx + 1]
                        substr3 = substr[idx + 1:]
                        break
                    else:  # "(abc)+", "(abc)*"
                        substr2 = substr[:idx + 2]
                        substr3 = substr[idx + 2:]
                        break
        elif substr[idx] == '[':
            if not square_brackets:
                square_brackets += 1
        elif substr[idx] == ']' and not substr[idx - 1] == '\\':
            square_brackets -= 1
            if not square_brackets and not round_brackets:
                if idx == len(substr) - 1:  # "(abc)":  # "[abc]"
                    substr2 = substr[:idx + 1]
                    substr3 = substr[idx + 1:]
                    break
                else:  # "[abc]+", "[abc]*"
                    substr2 = substr[:idx + 2]
                    substr3 = substr[idx + 2:]
                    break
        idx += 1

    if not substr3:
        f3(substr, A, B, matrix)
    else:
        K = next(C)
        f3(substr2, A, K, matrix)
        f2(substr3, K, B, matrix)


def f3(substr: str, A: str, B: str, matrix: dict) -> None:
    """ Formation of the "state matrix". """
    last_two_symbols = ''
    K = next(C)
    if len(substr) >= 2:
        last_two_symbols = ''.join([substr[-2], substr[-1]])
    # if substr in ALPHABET or substr == 'eps' or \
    if substr in ALPHABET or \
            (substr[0] == '\\' and
             substr[1] in SPECIALS and len(substr) == 2):  # 'a', 'eps', '\['
        matrix_add(substr, A, B, matrix)
    elif (substr[0] in ALPHABET and substr[-1] == '+') or \
            (substr[0] == '\\' and substr[-1] == '+'):  # 'a+', '\)+'
        pos_plus = substr.find('+')
        matrix_add(substr[:pos_plus], A, K, matrix)
        matrix_add(substr[:pos_plus], K, K, matrix)
        matrix_add('eps', K, B, matrix)
    elif (substr[0] in ALPHABET and substr[-1] == '*') or \
            (substr[0] == '\\' and substr[-1] == '*'):  # 'a*', '\)*'
        pos_mult = substr.find('*')
        matrix_add('eps', A, K, matrix)
        matrix_add(substr[:pos_mult], K, K, matrix)
        matrix_add('eps', K, B, matrix)
    elif substr[0] == '(' and substr[-1] == ')':  # ()
        f1(substr[1:-1], A, B, matrix)
    elif substr[0] == '(' and last_two_symbols == ")+":  # ()+
        f1(substr[1:-2], A, K, matrix)
        f1(substr[1:-2], K, K, matrix)
        matrix_add('eps', K, B, matrix)
    elif substr[0] == '(' and last_two_symbols == ")*":  # ()*
        matrix_add('eps', A, K, matrix)
        f1(substr[1:-2], K, K, matrix)
        matrix_add('eps', K, B, matrix)
    elif substr[0] == '[' and substr[-1] == ']':  # []
        new_substr = add_vertical_lines(substr[1:-1])
        f1(new_substr, A, B, matrix)
    elif substr[0] == '[' and last_two_symbols == "]+":  # []+
        new_substr = add_vertical_lines(substr[1:-2])
        new_substr = '(' + new_substr + ')+'
        f1(new_substr, A, B, matrix)
    elif substr[0] == '[' and last_two_symbols == "]*":  # []+
        new_substr = add_vertical_lines(substr[1:-2])
        new_substr = '(' + new_substr + ')*'
        f1(new_substr, A, B, matrix)


def add_vertical_lines(substr: str) -> str:
    """ Add vertical lines between each character. """
    tokens = []
    idx = 0
    while idx < len(substr):
        char = substr[idx]
        if char == '\\' and substr[idx + 1:idx + 2] in SPECIALS:
            tokens.append(substr[idx:idx + 2])
            idx += 1
        elif char in SPECIALS:
            tokens.append(f"\\{char}")
        else:
            tokens.append(char)
        idx += 1
    return '|'.join(tokens)


def matrix_add(
        substr: str,
        A: str,
        B: (str or list or bool),
        matrix: dict) -> None:
    """
    Adding an element to the matrix.*
    Type 'B' can be a "list", a "string" or "bool".
    """
    if not matrix.get(A):
        if isinstance(B, (list, bool)):
            matrix[A] = {substr: B}
        elif isinstance(B, str):
            matrix[A] = {substr: [B]}
        else:
            raise TypeError
    else:
        if isinstance(B, bool):
            matrix[A][substr] = B
        elif isinstance(B, str):
            if not matrix[A].get(substr):
                matrix[A][substr] = [B]
            elif B not in matrix[A][substr]:
                matrix[A][substr].append(B)
        elif isinstance(B, list):
            if not matrix[A].get(substr):
                matrix[A][substr] = B
            else:
                unique = get_a_sorted_list_of_vertices(
                    set(matrix[A][substr] + B)
                )
                matrix[A][substr] = unique
        else:
            raise TypeError


def get_a_sorted_list_of_vertices(x: dict or list or set) -> list:
    """ Sorting the matrix by the name of states. """
    def help_sorting(lst: list or set) -> int:
        for el in lst:
            if len(el) >= 2:
                yield int(el[1:])
    nums = list()
    new_list = list()
    z_flag = False
    init = 'A'
    final = 'Z'
    if isinstance(x, dict):
        nums = sorted(list(help_sorting(x.keys())))
        new_list.append(init)  # initial vertex
    elif isinstance(x, (list, set)):
        if init in x:
            new_list.append(init)  # initial vertex
        if final in x:
            z_flag = True
        nums = sorted(list(help_sorting(x)))
    for num in nums:
        new_list.append('S' + str(num))
    if z_flag:
        new_list.append(final)
    return new_list


def translate(eps_ndfsm: dict, dfsm: dict, expr: str) -> None:
    """
    Translation of a non-deterministic FSM to a deterministic FSM.
    (Determination of the FSM).
    """
    alphabet = get_alphabet(expr)
    vertices = get_a_sorted_list_of_vertices(eps_ndfsm)
    remove_of_eps_edges(eps_ndfsm, vertices)
    # remove_of_eps_edges(eps_ndfsm, ndfsm, alphabet)
    remove_unreachable_states(eps_ndfsm, 'A')
    create_dfsm(eps_ndfsm, dfsm, alphabet)


def remove_of_eps_edges(
        d: dict,
        vertices: list) -> None:
    """
    Removing empty transitions in eps_ndFSM.
    ndFSM -- non-deterministic FSM.
    eps_ndFSM -- non-deterministic FSM with eps-transitions..
    """
    final = 'Z'
    for v in vertices:
        while d[v].get('eps'):
            eps_v = d[v]['eps'][0]
            if eps_v != final:
                d[v] = combining_dict(d[v], d[eps_v])
                d[v]['eps'].remove(eps_v)
                if not d[v].get('eps'):
                    del d[v]['eps']
            else:
                del d[v]['eps']


def combining_dict(d1: dict, d2: dict) -> dict:
    """ Combining the values of two dictionaries for a given key. """
    new_dict = dict()
    for d1_key in d1.keys():
        for d2_key in d2.keys():
            if d1_key == d2_key:
                content = get_a_sorted_list_of_vertices(
                    set(d1[d1_key]) | set(d2[d2_key])
                )
                new_dict[d1_key] = content
            else:
                if new_dict.get(d1_key):
                    new_dict[d1_key].extend(d1[d1_key])
                    new_dict[d1_key] = get_a_sorted_list_of_vertices(
                        set(new_dict[d1_key])
                    )
                else:
                    new_dict[d1_key] = get_a_sorted_list_of_vertices(
                        set(d1[d1_key])
                    )
                if new_dict.get(d2_key):
                    new_dict[d2_key].extend(d2[d2_key])
                    new_dict[d2_key] = get_a_sorted_list_of_vertices(
                        set(new_dict[d2_key])
                    )
                else:
                    new_dict[d2_key] = get_a_sorted_list_of_vertices(
                        set(d2[d2_key])
                    )
    return new_dict


def create_dfsm(d1: dict, d2: dict, alphabet: list) -> None:
    """
    Creation of a determinate finite automaton.
    Remove transitions on the same symbol.
    """

    """
    1) Создаем новые вершины для 'A'. 
    Цикл по всем символам из алфавита. 
    Если d_old['A'].get(symbol), то создаем вершину -- объединение входящих
    вершин в список. 
    Пример: A: {'x': ['S1', 'S3', 'Z']}. 
    Результат: новая вершина A: {'x': "S1_S3_Z"}.
    2) Просматриваем созданные вершины из А по каждому символу из алфавита.
    Заходим в каждую из них и смотрим, в какие вершины они вели по данному 
    символу. Создаем вершину -- объединение всех вершин, которые вели по 
    данному символу в другие вершины.
    Пример: A: {'x': ['S1', 'S5']}, S1: {'x': ['S2', 'S3']}, S5: {'x': ['S6']}.
    Результат: переход в новую вершину по 'x': "S1_S5": {'x': "S2_S3_S6"}.
    3) Если вершина уже была создана в ДКА, то в первую очередь смотрим, 
    возможно ли соединить эту вершину по данному символу с исходной.
    """
    init = 'A'
    final = 'Z'
    for s in d1[init].keys():
        split = d1[init][s]
        new_key = "_".join(split)
        matrix_add(s, init, new_key, d2)

    vertices_list = list(init)
    visited_vertices = list()
    while vertices_list:  # All vertices of d2
        v = vertices_list[0]
        del vertices_list[0]
        for s in d2[v].keys():  # All transition symbols from the vertex v
            # All vertices for which there is a transition on the symbol s
            for v2 in d2[v][s]:
                if v2 == final:
                    continue
                tmp_dict = dict()
                # All vertices from the "string" v2
                for v3 in v2.split('_'):
                    if v3 != final:
                        # All symbols of the alphabet
                        for s2 in alphabet:
                            if d1[v3].get(s2):
                                link = get_a_sorted_list_of_vertices(
                                    d1[v3][s2]
                                )
                                matrix_add(s2, v2, link, tmp_dict)
                for tmp_v in tmp_dict[v2].keys():
                    tmp_dict[v2][tmp_v] = ['_'.join(tmp_dict[v2][tmp_v])]
                d2[v2] = tmp_dict[v2]
                if v2 not in visited_vertices:
                    visited_vertices.append(v2)
                    vertices_list.append(v2)


def get_reachable_states(d: dict, node: str, visited: set = set()) -> set:
    """ Obtaining a set of reachable states. """
    states = set()
    if d.get(node):
        for symbol in d[node].keys():
            if symbol != 'eps':
                states_list = d[node][symbol]
                for state in states_list:
                    if state not in states \
                            and state not in visited \
                            and state != node:
                        rec_nodes = get_reachable_states(d, state, states)
                        states = states | rec_nodes
                    states.add(state)
    return states


def remove_unreachable_states(d: dict, start_state: str) -> None:
    """ Removing states that are not connected to any vertex. """
    states = get_reachable_states(d, start_state)
    states.add(start_state)
    for node in d.keys() - states:
        del d[node]


# def remove_of_eps_edges(
#         eps_ndfsm: dict,
#         ndfsm: dict,
#         alphabet: list) -> None:
#     """
#     Removing empty transitions in eps_ndFSM.
#     eps_ndFSM -- non-deterministic FSM with eps-transitions;
#     ndFSM -- non-deterministic FSM.
#     Here we use the Deep-First Search algorithm.
#     """
#     final = 'Z'
#     for v in eps_ndfsm.keys():
#         flag = False
#         for s in alphabet:
#             v_list = list()
#             visited = list()
#             new_v_list = list(
#                 set(search_vertices(eps_ndfsm, v, s, v_list, visited))
#             )
#             if new_v_list:
#                 matrix_add(s, v, new_v_list, ndfsm)
#                 if final in new_v_list:
#                     flag = True
#             if eps_ndfsm[v].get('end') and eps_ndfsm[v]['end']:
#                 flag = True
#         matrix_add('end', v, flag, ndfsm)
#
#
# def search_vertices(
#         d: dict,
#         vertex: str,
#         s: str,
#         v_list: list,
#         visited: list) -> list:
#     """
#     Search for the vertices with which the current vertex is bound by
#     epsilon transitions or by the symbol 's'.
#     """
#     final = 'Z'
#     if vertex == final:
#         return
#
#     if d[vertex].get(s) and final in d[vertex][s]:
#         matrix_add('end', vertex, True, d)
#     elif d[vertex].get('eps') and final in d[vertex]['eps']:
#         matrix_add('end', vertex, True, d)
#
#     if d[vertex].get(s):  # search by 's'
#         v_list.extend(d[vertex].get(s))
#         v_list = list(set(v_list))
#         for v in v_list:
#             if v not in visited:
#                 visited.append(v)
#                 search_vertices(d, v, s, v_list, visited)
#     if d[vertex].get('eps'):  # search by 'eps'
#         for v in d[vertex]['eps']:
#             if v not in visited:
#                 visited.append(v)
#                 if d.get(v) and d[v].get(s):  # search by 's'
#                     v_list.extend(d[v].get(s))
#                     v_list = list(set(v_list))
#                 search_vertices(d, v, s, v_list, visited)
#     return v_list


def print_matrix(x: dict) -> None:
    """ Display the matrix. """
    max_length = len(max(x.keys(), key=len))
    for s, vers in x.items():
        curr_length = len(s)
        for _ in range(max_length - curr_length):
            print(' ', sep='', end='')
        print(s, ': ', sep='', end='')
        print(vers, sep='')


def get_alphabet(expr: str) -> list:
    """
    Getting the alphabet of a regular expression.*
    *For example: r"(a+2)*b" returns ['a', '2', 'b'].
    """
    idx = 0
    characters = list()
    while idx < len(expr):
        if expr[idx] in ALPHABET and expr[idx] not in characters:
            characters.append(expr[idx])
        elif idx < len(expr) - 1 and expr[idx] == '\\' and \
                expr[idx + 1] in SPECIALS and \
                expr[idx:idx + 2] not in characters:
            characters.append(expr[idx:idx + 2])
            idx += 1
        idx += 1
    return characters


def print_graph(d: dict, name: str) -> None:
    """ Displays the graph using graphviz. """
    g = gv.Digraph(name, None, name, None, 'pdf')
    for vertex, subdict in d.items():
        g.node(vertex)
        for symbol in subdict.keys():
            v_list = d[vertex][symbol]
            for v in v_list:
                g.node(v)
                g.edge(vertex, v, symbol)

    g.render(filename=name, directory='output-files', view=False)


if __name__ == "__main__":
    C = states_generator(0)
    I, Z = 'A', 'Z'  # initial and final states
    eps_ndfsm_matrix = dict()
    dfsm_matrix = dict()
    regexp = r"(xy*|ab|(x|a*))(x|y*)"  # regular expression
    # regexp = r"y*|xy|x*y|x+"
    # regexp = r"(a|b)|(c|d)*"
    # regexp = r"xy|xz|xyz|xt|(a+x)*"
    f1(regexp, I, Z, eps_ndfsm_matrix)
    print("\neps-NDFSM")
    print_matrix(eps_ndfsm_matrix)
    print_graph(eps_ndfsm_matrix, 'eps-NDFSM')
    translate(eps_ndfsm_matrix, dfsm_matrix, regexp)
    print("\nNDFSM")
    print_matrix(eps_ndfsm_matrix)
    print_graph(eps_ndfsm_matrix, 'NDFSM')
    print("\nDFSM")
    print_matrix(dfsm_matrix)
    print_graph(dfsm_matrix, 'DFSM')
