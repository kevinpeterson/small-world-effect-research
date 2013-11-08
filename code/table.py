import numpy as np
from collections import OrderedDict

summary_functions = OrderedDict(
    [("Mean", lambda x : round(np.mean(x),3)),
     ("Median", lambda x : round(np.median(x),3)),
     ("Max", lambda x : round(np.max(x),3)),
     ("Min", lambda x : round(np.min(x),3)),
     ("Std", lambda x : round(np.std(x),3))]
)

def create_table(title,label,N,stats,functions=summary_functions):
    table = ""
    table += "\\begin{table}[htbp]\centering\n"
    table += "\caption{\label{"+label+"}\n"
    table += "\\textbf{Statistics} }\\begin{tabular} {@{} l r "+ "".join(" r " for _ in range(len(functions))) +" @{}} \\\ \hline\n"
    table += "\\textbf{Statistic} & " + " & ".join(["\\textbf{%s}" % key for key in functions.keys()]) + " \\\ \n"
    table += "\hline\n"
    table += "".join([s + " & " + " & ".join([str(f(stats[s])) for k,f in functions.items()]) + " \\\ \n" for s in stats])
    table += "\hline\n"
    table += "\multicolumn{"+str(len(functions)+1)+"}{@{}l}{\n"
    table += "N = "+str(N)+"}\n"
    table += "\end{tabular}\n"
    table += "\end{table}\n"

    return table