def matrix_to_table(matrix, caption):
    latex_str = "\n \\begin{table}[H] \centering \\begin{tabular}{c|c|c|c|c|c} \diagbox{$y$}{$x$} & $\\bm{0}$ & $\\bm{1}$ & $\\bm{2}$ & $\\bm{3}$ & $\\bm{4}$"
    for row_nb, row in enumerate(matrix):
        latex_str += " \\\\ \hline\n"
        latex_str += "$\\bm{{{}}} $".format(row_nb)
        for element in row:
            latex_str += " & \\num{{{0:.3f}}}".format(element)

    latex_str += "\\ \n \end{{{}}} \n \\caption{{{}}} \n \label{{{}}} \n \end{{{}}}".format("tabular", caption, "tab:j.values.deterministic", "table")

    return latex_str

def matrix_to_table_string(matrix, caption):
    latex_str = "\n \\begin{table}[H] \centering \\begin{tabular}{c|c|c|c|c|c} \diagbox{$y$}{$x$} & $\\bm{0}$ & $\\bm{1}$ & $\\bm{2}$ & $\\bm{3}$ & $\\bm{4}$"
    for row_nb, row in enumerate(matrix):
        latex_str += " \\\\ \hline\n"
        latex_str += "$\\bm{{{}}} $".format(row_nb)
        for element in row:
            latex_str += " & {}".format(element)

    latex_str += "\\ \n \end{{{}}} \n \\caption{{{}}} \n \label{{{}}} \n \end{{{}}}".format("tabular", caption, "tab:j.values.deterministic", "table")

    return latex_str