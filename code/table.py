import numpy as np
from collections import OrderedDict

sigdigits = 2

summary_functions = OrderedDict(
    [("Mean", lambda x : round(np.mean(x),sigdigits)),
     ("Med", lambda x : round(np.median(x),sigdigits)),
     ("Max", lambda x : round(np.max(x),sigdigits)),
     ("Min", lambda x : round(np.min(x),sigdigits)),
     ("Std", lambda x : round(np.std(x),sigdigits))]
)

def create_table(title,label,caption,stats,functions=summary_functions):
    table = ""
    table += "\\begin{table}[htbp]\centering\n"
    table += "\caption{\label{"+label+"}\n"
    table += "\\textbf{Statistics} }\\begin{tabular} {@{} l r "+ "".join(" r " for _ in range(len(functions))) +" @{}} \\\ \hline\n"
    table += "\\textbf{Statistic} & " + " & ".join(["\\textbf{%s}" % key for key in functions.keys()]) + " \\\ \n"
    table += "\hline\n"
    table += "".join([s + " & " + " & ".join([str(f(stats[s])) for k,f in functions.items()]) + " \\\ \n" for s in stats])
    table += "\hline\n"
    table += "\multicolumn{"+str(len(functions)+1)+"}{@{}l}{\n"
    table += caption+"}\n"
    table += "\end{tabular}\n"
    table += "\end{table}\n"

    return table