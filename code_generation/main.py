# Translator of mathematical expressions into code in Assembler language.
from pythonds.basic.stack import Stack
from pythonds.trees.binaryTree import BinaryTree
from graphviz import Digraph
from sys import argv
from re import match

OPERATORS = frozenset("+-*/") | {'mul', 'add', 'sub', 'div'}
DIGITS = frozenset("0123456789")
ALPHAS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_")
ASSIGNMENT = frozenset("=")
COMMUTATIVE = frozenset("+*") | {'add', 'mul'}
NONCOMMUTATIVE = frozenset("-/") | {'sub', 'div'}


class BadExpressionError(BaseException):
    pass


class SimpleBinaryTree:
    """ Implementation of a simple BinaryTree class. """

    def __init__(self, head, left=None, right=None):
        self.head = head
        self.left = left
        self.right = right

    def __str__(self):
        return str(self.head)


class MathExprParser:
    @staticmethod
    def tokenize(math_expr: str) -> list:
        """
        Split a string into tokens.
        Tokens: operators (['+', '-', '*', '/']), '(', ')', unsigned numbers,
        variables (it can start with a latin symbol and end with a latin symbol
        or a number).
        :param math_expr: input math expression
        :return: list of tokens
        """
        tokens = list()
        idx = 0
        math_expr = math_expr.replace(' ', '')
        while idx < len(math_expr):
            symbol = math_expr[idx]
            if symbol in OPERATORS or symbol in ('(', '=', ')'):
                tokens.append(symbol)
            elif symbol in DIGITS:
                number = str()
                while math_expr[idx:idx + 1] in DIGITS:
                    number += math_expr[idx]
                    idx += 1
                tokens.append(number)
                continue
            elif symbol in ALPHAS:
                variable = str()
                while math_expr[idx:idx + 1] in ALPHAS | DIGITS:
                    variable += math_expr[idx]
                    idx += 1
                tokens.append(variable)
                continue
            else:
                err_string = "Illegal symbol in the string. ({} in pos {})."
                raise BadExpressionError(err_string.format(symbol, idx))
            idx += 1
        return tokens

    @staticmethod
    def build_bad_parse_tree_with_parentheses(math_expr: str) -> BinaryTree:
        """
        Building a tree according to the incoming mathematical expression with
        the operators "+, -, *, /" and with parentheses.
        All expressions should start with "<var_name> = ...", where <var_name>
        is the name of the variable.
        Since this is a binary tree, each expression must be placed in
        parentheses.
        :param math_expr: input math expression
        :return: binary tree
        """
        tokens = MathExprParser.tokenize(math_expr)
        stack = Stack()
        tree = BinaryTree('')
        tree_head = tree
        tree.setRootVal(tokens[1])
        tree.insertLeft(tokens[0])
        tree.insertRight('')
        tree = tree.rightChild
        stack.push(tree)
        for token in tokens[2:]:
            if token == '(':
                tree.insertLeft('')
                stack.push(tree)
                tree = tree.getLeftChild()
            elif token not in OPERATORS and token != ')':
                tree.setRootVal(token)
                parent = stack.pop()
                tree = parent
            elif token in OPERATORS:
                tree.setRootVal(token)
                tree.insertRight('')
                stack.push(tree)
                tree = tree.getRightChild()
            elif token == ')':
                tree = stack.pop()
            else:
                err_string = f"The character {token} can not be recognized."
                raise BadExpressionError(err_string)
        return tree_head

    @staticmethod
    def build_parse_tree(math_expr: str) -> SimpleBinaryTree:
        """
        Construction of left binary tree according to left given mathematical
        expression.
        :param math_expr: input math expression
        :return: binary tree
        """
        tokens = MathExprParser.tokenize(math_expr)
        tokens.append('end')

        def get_token(token_list: list, expected_symbol: str) -> bool:
            condition = token_list[0] == expected_symbol
            if condition:
                del token_list[0]
            return condition

        def get_sum(token_list: list) -> SimpleBinaryTree:
            left = get_sub(token_list)
            if get_token(token_list, '+'):
                right = get_sum(token_list)
                return SimpleBinaryTree('+', left, right)
            else:
                return left

        def get_sub(token_list: list) -> SimpleBinaryTree:
            left = get_product(token_list)
            if get_token(token_list, '-'):
                right = get_sub(token_list)
                return SimpleBinaryTree('-', left, right)
            else:
                return left

        def get_product(token_list: list) -> SimpleBinaryTree:
            left = get_div(token_list)
            if get_token(token_list, '*'):
                right = get_product(token_list)
                return SimpleBinaryTree('*', left, right)
            else:
                return left

        def get_div(token_list: list) -> SimpleBinaryTree:
            left = get_node(token_list)
            if get_token(token_list, '/'):
                right = get_div(token_list)
                return SimpleBinaryTree('/', left, right)
            else:
                return left

        def get_node(token_list: list) -> SimpleBinaryTree:
            if get_token(token_list, '('):
                node = get_sum(token_list)
                if not get_token(token_list, ')'):
                    raise BadExpressionError('missing parentheses')
                return node
            else:
                node = token_list[0]
                token_list[:] = token_list[1:]
                return SimpleBinaryTree(node)

        if tokens[1] == '=':
            simple_binary_tree = SimpleBinaryTree('=')
            simple_binary_tree.left = SimpleBinaryTree(tokens[0])
            simple_binary_tree.right = get_sum(tokens[2:])
        else:
            simple_binary_tree = get_sum(tokens)
        return simple_binary_tree

    @staticmethod
    def print_tree(tree: (BinaryTree, SimpleBinaryTree),
                   level: int = 1) -> None:
        """
        Output the tree to the screen.
        Don't change the level indentation parameter 'level'.
        :param tree: top of input binary tree
        :param level: indentation parameter
        :return:
        """
        if tree:
            if isinstance(tree, BinaryTree):
                MathExprParser.print_tree(tree.getRightChild(), level + 1)
                for i in range(level):
                    print(5 * " ", end='', sep='')
                print(tree.getRootVal())
                MathExprParser.print_tree(tree.getLeftChild(), level + 1)
            elif isinstance(tree, SimpleBinaryTree):
                MathExprParser.print_tree(tree.right, level + 1)
                for i in range(level):
                    print(5 * " ", end='', sep='')
                print(tree.head)
                MathExprParser.print_tree(tree.left, level + 1)
            else:
                raise TypeError

    @staticmethod
    def print_tree_postorder(tree: (BinaryTree, SimpleBinaryTree)) -> None:
        if tree:
            if isinstance(tree, BinaryTree):
                MathExprParser.print_tree_postorder(tree.getRightChild())
                MathExprParser.print_tree_postorder(tree.getLeftChild())
                print(tree.getRootVal(), end=' ', sep='')
            elif isinstance(tree, SimpleBinaryTree):
                MathExprParser.print_tree_postorder(tree.right)
                MathExprParser.print_tree_postorder(tree.left)
                print(tree.head, end=' ', sep='')

    @staticmethod
    def draw_tree(tree: SimpleBinaryTree, math_expr: str) -> None:
        """
        Drawing a tree using the GraphViz tool.
        :param math_expr:
        :param tree: top of input binary tree
        :return: None
        """
        graph = Digraph(name='Tree', format='pdf')

        def generator(index: int = 0) -> str:
            """
            Generation of unique names of vertices.
            :param index: initial index
            :return: a string with a unique index
            """
            while True:
                index += 1
                yield f"V{index}"

        gen = generator()

        def connecting_vertices(
                g: Digraph,
                t: SimpleBinaryTree,
                top: str = None) -> None:
            """
            The connection of the vertices of the graph with each other.
            :param top: additional parameter to connect head vertices in graph
            with each other.
            Example: tree.left.head is connected to tree.head when a
            recursive call of the left or right child occurs.
            :param g: input Digraph
            :param t: input simple BinaryTree
            :return:
            """
            nonlocal gen
            if top is None:
                head = 'Head ' + next(gen)
                g.node(head, label=t.head)
            else:
                head = top
            left = 'Left ' + next(gen)
            right = 'Right ' + next(gen)
            if t.left:
                g.node(left, label=t.left.head)
                g.edge(head, left)
                connecting_vertices(g, t.left, left)
            if t.right:
                g.node(right, label=t.right.head)
                g.edge(head, right)
                connecting_vertices(g, t.right, right)

        connecting_vertices(graph, tree)

        graph.attr(label="\n\n" + math_expr, fontsize='16')
        graph.node_attr.update(shape='ellipse')
        # more shapes: https://graphviz.gitlab.io/_pages/doc/info/shapes.html
        # more examples: http://graphviz.readthedocs.io/en/stable/examples.html
        graph.render('tree', directory='output-files', view=False)


code = ''


def generate_code(tree: SimpleBinaryTree) -> str:
    """
    Generating code in the Assembler language for the input binary tree.
    :type tree: input simple BinaryTree
    :return: code of mathematical expression in the language of Assembler
    """
    global code

    def is_constant(string_obj: str) -> bool:
        """
        Check for whether the input object is a number or whether it
        corresponds to a variable declaration.
        :param string_obj: input string
        :return: True if 'obj' is unsigned number or variable
        """
        matching1 = match(r'\d+', string_obj)
        matching2 = match(r'[A-z_][A-z0-9_]*', string_obj)
        return matching1 or matching2

    def convert_operator(op: str) -> str:
        """
        Convert incoming operator to its string representation.
        :param op: input operator
        :return: string representation of the incoming operator
        """
        op_dict = {
            '+': 'add',
            '-': 'sub',
            '*': 'mul',
            '/': 'div'
        }
        return op_dict[op]

    if tree.head in ASSIGNMENT:
        """ 
        Variable assignment
        tree.head - assignment operator 
        tree.left - variable
        tree.right - subtree or a constant value or a unsigned integer
        """
        var_name = tree.left.head
        right_subtree = tree.right
        right_const = tree.right.head
        if is_constant(right_const):
            """ tree.right - a constant value or a unsigned integer """
            code += f'mov EAX, {right_const}\n'
        elif isinstance(right_subtree, SimpleBinaryTree):
            """ tree.right - subtree """
            code += generate_code(right_subtree)
        code += f'mov {var_name}, EAX\n'
    elif tree.head in OPERATORS:
        """ tree.head - binary operator """
        operator = convert_operator(tree.head)
        if is_constant(tree.left.head) and is_constant(tree.right.head):
            """
            tree.left - a constant value or a unsigned integer
            tree.right - a constant value or a unsigned integer
            """
            const_left = tree.left.head
            const_right = tree.right.head
            code += f'mov EAX, {const_left}\n'
            code += f'{operator} EAX, {const_right}\n'
        elif is_constant(tree.right.head) and \
                isinstance(tree.left.head, SimpleBinaryTree):
            """
            tree.left - subtree
            tree.right - a constant value or a unsigned integer
            """
            left_subtree = tree.left
            const_right = tree.right.head
            code += generate_code(left_subtree)
            code += f'{operator} EAX, {const_right}'
        elif operator in COMMUTATIVE and \
                isinstance(tree.right, SimpleBinaryTree) and \
                is_constant(tree.left.head):
            """
            tree.head - commutative operator
            tree.left - a constant value or a unsigned integer
            tree.right - subtree
            """
            right_subtree = tree.right
            const_left = tree.left.head
            generate_code(right_subtree)
            code += f'{operator} EAX, {const_left}\n'
        elif operator in NONCOMMUTATIVE and \
                isinstance(tree.right, SimpleBinaryTree) and \
                is_constant(tree.left.head):
            """
            tree.head - non-commutative operator
            tree.left - a constant value or a unsigned integer
            tree.right - subtree            
            """
            const_left = tree.left.head
            right_subtree = tree.right
            generate_code(right_subtree)
            code += f'mov EDX, {const_left}\n'
            code += 'xchg EAX, EDX\n'
            code += f'{operator} EAX, EDX\n'
        elif isinstance(tree.left, SimpleBinaryTree) and \
                isinstance(tree.right, SimpleBinaryTree):
            generate_code(tree.right)
            code += 'push EAX\n'
            generate_code(tree.left)
            code += 'pop EDX\n'
            code += f'{operator} EAX, EDX\n'
    return code


def write_code_to_file(math_expr: str, g_code: str, filename: str) -> None:
    """
    Writing code to file.
    :param math_expr: input math expression
    :param g_code: generated code in Assembler language
    :param filename: file name to which the code will be saved
    """
    if filename[:-4] != '.txt':
        filename += '.txt'

    if filename[:13] != 'output-files/':
        filename = 'output-files/' + filename

    with open(filename, 'w') as f:
        f.write(math_expr + '\n')
        f.write(g_code)


if __name__ == '__main__':
    args = argv[1:]
    if args and args[0] in ('-i', '--i'):
        # interactive mode
        expr = input("Enter math expression -> ")
    else:
        # program mode
        expr = "A = 2 + 3 - 5"
    inst = MathExprParser()
    parse_tree = inst.build_parse_tree(expr)
    gen_code = generate_code(parse_tree)
    write_code_to_file(expr, gen_code, 'ass_code')
    print(expr)
    print(gen_code)
    inst.draw_tree(parse_tree, expr)
