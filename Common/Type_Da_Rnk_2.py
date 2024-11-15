from typing import Literal as Lit


TP = Lit[
    'High LS Rtn Rank',
    'High LS Rtn +1D',
    'High LS Rtn +3D',
    'High LS Rtn +5D',
    'High LS Rtn +10D',
    'High LS Rtn +20D',
    'High LS Rtn +40D',
    'High Cor Rank',
    'High Cor 1D',
    'High Cor 3D',
    'High Cor 5D',
    'High Cor 10D',
    'High Cor 20D',
    'High Cor 40D',
    'High Rank',
    'Low Rank',
    'High Gap',
    'Low Gap',
    'High Gap 1D',
    'Low Gap 1D',
    'High Gap 3D',
    'Low Gap 3D',
    'High Gap 5D',
    'Low Gap 5D',
    'High Gap 10D',
    'Low Gap 10D',
    'High Gap 20D',
    'Low Gap 20D',
    'High Gap 40D',
    'Low Gap 40D',
    'High Gap Avg',
    'Low Gap Avg',
    'High Gap.1 Avg.1',
    'Low Gap.1 Avg.1',
    'High Gap.1 Avg.3',
    'Low Gap.1 Avg.3',
    'High Gap.1 Avg.5',
    'Low Gap.1 Avg.5',
    'High Gap.1 Avg.10',
    'Low Gap.1 Avg.10',
    'High Gap.1 Avg.20',
    'Low Gap.1 Avg.20',
    'High Gap.1 Avg.40',
    'Low Gap.1 Avg.40',
    'High Gap.3 Avg.1',
    'Low Gap.3 Avg.1',
    'High Gap.3 Avg.3',
    'Low Gap.3 Avg.3',
    'High Gap.3 Avg.5',
    'Low Gap.3 Avg.5',
    'High Gap.3 Avg.10',
    'Low Gap.3 Avg.10',
    'High Gap.3 Avg.20',
    'Low Gap.3 Avg.20',
    'High Gap.3 Avg.40',
    'Low Gap.3 Avg.40',
    'High Gap.5 Avg.1',
    'Low Gap.5 Avg.1',
    'High Gap.5 Avg.3',
    'Low Gap.5 Avg.3',
    'High Gap.5 Avg.5',
    'Low Gap.5 Avg.5',
    'High Gap.5 Avg.10',
    'Low Gap.5 Avg.10',
    'High Gap.5 Avg.20',
    'Low Gap.5 Avg.20',
    'High Gap.5 Avg.40',
    'Low Gap.5 Avg.40',
    'High Gap.10 Avg.1',
    'Low Gap.10 Avg.1',
    'High Gap.10 Avg.3',
    'Low Gap.10 Avg.3',
    'High Gap.10 Avg.5',
    'Low Gap.10 Avg.5',
    'High Gap.10 Avg.10',
    'Low Gap.10 Avg.10',
    'High Gap.10 Avg.20',
    'Low Gap.10 Avg.20',
    'High Gap.10 Avg.40',
    'Low Gap.10 Avg.40',
    'High Gap.20 Avg.1',
    'Low Gap.20 Avg.1',
    'High Gap.20 Avg.3',
    'Low Gap.20 Avg.3',
    'High Gap.20 Avg.5',
    'Low Gap.20 Avg.5',
    'High Gap.20 Avg.10',
    'Low Gap.20 Avg.10',
    'High Gap.20 Avg.20',
    'Low Gap.20 Avg.20',
    'High Gap.20 Avg.40',
    'Low Gap.20 Avg.40',
    'High Gap.40 Avg.1',
    'Low Gap.40 Avg.1',
    'High Gap.40 Avg.3',
    'Low Gap.40 Avg.3',
    'High Gap.40 Avg.5',
    'Low Gap.40 Avg.5',
    'High Gap.40 Avg.10',
    'Low Gap.40 Avg.10',
    'High Gap.40 Avg.20',
    'Low Gap.40 Avg.20',
    'High Gap.40 Avg.40',
    'Low Gap.40 Avg.40',
    'High Gap Dev',
    'Low Gap Dev',
    'High Gap.1 Dev.1',
    'Low Gap.1 Dev.1',
    'High Gap.1 Dev.3',
    'Low Gap.1 Dev.3',
    'High Gap.1 Dev.5',
    'Low Gap.1 Dev.5',
    'High Gap.1 Dev.10',
    'Low Gap.1 Dev.10',
    'High Gap.1 Dev.20',
    'Low Gap.1 Dev.20',
    'High Gap.1 Dev.40',
    'Low Gap.1 Dev.40',
    'High Gap.3 Dev.1',
    'Low Gap.3 Dev.1',
    'High Gap.3 Dev.3',
    'Low Gap.3 Dev.3',
    'High Gap.3 Dev.5',
    'Low Gap.3 Dev.5',
    'High Gap.3 Dev.10',
    'Low Gap.3 Dev.10',
    'High Gap.3 Dev.20',
    'Low Gap.3 Dev.20',
    'High Gap.3 Dev.40',
    'Low Gap.3 Dev.40',
    'High Gap.5 Dev.1',
    'Low Gap.5 Dev.1',
    'High Gap.5 Dev.3',
    'Low Gap.5 Dev.3',
    'High Gap.5 Dev.5',
    'Low Gap.5 Dev.5',
    'High Gap.5 Dev.10',
    'Low Gap.5 Dev.10',
    'High Gap.5 Dev.20',
    'Low Gap.5 Dev.20',
    'High Gap.5 Dev.40',
    'Low Gap.5 Dev.40',
    'High Gap.10 Dev.1',
    'Low Gap.10 Dev.1',
    'High Gap.10 Dev.3',
    'Low Gap.10 Dev.3',
    'High Gap.10 Dev.5',
    'Low Gap.10 Dev.5',
    'High Gap.10 Dev.10',
    'Low Gap.10 Dev.10',
    'High Gap.10 Dev.20',
    'Low Gap.10 Dev.20',
    'High Gap.10 Dev.40',
    'Low Gap.10 Dev.40',
    'High Gap.20 Dev.1',
    'Low Gap.20 Dev.1',
    'High Gap.20 Dev.3',
    'Low Gap.20 Dev.3',
    'High Gap.20 Dev.5',
    'Low Gap.20 Dev.5',
    'High Gap.20 Dev.10',
    'Low Gap.20 Dev.10',
    'High Gap.20 Dev.20',
    'Low Gap.20 Dev.20',
    'High Gap.20 Dev.40',
    'Low Gap.20 Dev.40',
    'High Gap.40 Dev.1',
    'Low Gap.40 Dev.1',
    'High Gap.40 Dev.3',
    'Low Gap.40 Dev.3',
    'High Gap.40 Dev.5',
    'Low Gap.40 Dev.5',
    'High Gap.40 Dev.10',
    'Low Gap.40 Dev.10',
    'High Gap.40 Dev.20',
    'Low Gap.40 Dev.20',
    'High Gap.40 Dev.40',
    'Low Gap.40 Dev.40',
    'High Gap Lvl',
    'Low Gap Lvl',
    'High Gap.1 Lvl.1',
    'Low Gap.1 Lvl.1',
    'High Gap.1 Lvl.3',
    'Low Gap.1 Lvl.3',
    'High Gap.1 Lvl.5',
    'Low Gap.1 Lvl.5',
    'High Gap.1 Lvl.10',
    'Low Gap.1 Lvl.10',
    'High Gap.1 Lvl.20',
    'Low Gap.1 Lvl.20',
    'High Gap.1 Lvl.40',
    'Low Gap.1 Lvl.40',
    'High Gap.3 Lvl.1',
    'Low Gap.3 Lvl.1',
    'High Gap.3 Lvl.3',
    'Low Gap.3 Lvl.3',
    'High Gap.3 Lvl.5',
    'Low Gap.3 Lvl.5',
    'High Gap.3 Lvl.10',
    'Low Gap.3 Lvl.10',
    'High Gap.3 Lvl.20',
    'Low Gap.3 Lvl.20',
    'High Gap.3 Lvl.40',
    'Low Gap.3 Lvl.40',
    'High Gap.5 Lvl.1',
    'Low Gap.5 Lvl.1',
    'High Gap.5 Lvl.3',
    'Low Gap.5 Lvl.3',
    'High Gap.5 Lvl.5',
    'Low Gap.5 Lvl.5',
    'High Gap.5 Lvl.10',
    'Low Gap.5 Lvl.10',
    'High Gap.5 Lvl.20',
    'Low Gap.5 Lvl.20',
    'High Gap.5 Lvl.40',
    'Low Gap.5 Lvl.40',
    'High Gap.10 Lvl.1',
    'Low Gap.10 Lvl.1',
    'High Gap.10 Lvl.3',
    'Low Gap.10 Lvl.3',
    'High Gap.10 Lvl.5',
    'Low Gap.10 Lvl.5',
    'High Gap.10 Lvl.10',
    'Low Gap.10 Lvl.10',
    'High Gap.10 Lvl.20',
    'Low Gap.10 Lvl.20',
    'High Gap.10 Lvl.40',
    'Low Gap.10 Lvl.40',
    'High Gap.20 Lvl.1',
    'Low Gap.20 Lvl.1',
    'High Gap.20 Lvl.3',
    'Low Gap.20 Lvl.3',
    'High Gap.20 Lvl.5',
    'Low Gap.20 Lvl.5',
    'High Gap.20 Lvl.10',
    'Low Gap.20 Lvl.10',
    'High Gap.20 Lvl.20',
    'Low Gap.20 Lvl.20',
    'High Gap.20 Lvl.40',
    'Low Gap.20 Lvl.40',
    'High Gap.40 Lvl.1',
    'Low Gap.40 Lvl.1',
    'High Gap.40 Lvl.3',
    'Low Gap.40 Lvl.3',
    'High Gap.40 Lvl.5',
    'Low Gap.40 Lvl.5',
    'High Gap.40 Lvl.10',
    'Low Gap.40 Lvl.10',
    'High Gap.40 Lvl.20',
    'Low Gap.40 Lvl.20',
    'High Gap.40 Lvl.40',
    'Low Gap.40 Lvl.40',
    'High Abs Gap',
    'Low Abs Gap',
    'High Abs Gap 1D',
    'Low Abs Gap 1D',
    'High Abs Gap 3D',
    'Low Abs Gap 3D',
    'High Abs Gap 5D',
    'Low Abs Gap 5D',
    'High Abs Gap 10D',
    'Low Abs Gap 10D',
    'High Abs Gap 20D',
    'Low Abs Gap 20D',
    'High Abs Gap 40D',
    'Low Abs Gap 40D',
    'High Abs Gap Avg',
    'Low Abs Gap Avg',
    'High Abs Gap.1 Avg.1',
    'Low Abs Gap.1 Avg.1',
    'High Abs Gap.1 Avg.3',
    'Low Abs Gap.1 Avg.3',
    'High Abs Gap.1 Avg.5',
    'Low Abs Gap.1 Avg.5',
    'High Abs Gap.1 Avg.10',
    'Low Abs Gap.1 Avg.10',
    'High Abs Gap.1 Avg.20',
    'Low Abs Gap.1 Avg.20',
    'High Abs Gap.1 Avg.40',
    'Low Abs Gap.1 Avg.40',
    'High Abs Gap.3 Avg.1',
    'Low Abs Gap.3 Avg.1',
    'High Abs Gap.3 Avg.3',
    'Low Abs Gap.3 Avg.3',
    'High Abs Gap.3 Avg.5',
    'Low Abs Gap.3 Avg.5',
    'High Abs Gap.3 Avg.10',
    'Low Abs Gap.3 Avg.10',
    'High Abs Gap.3 Avg.20',
    'Low Abs Gap.3 Avg.20',
    'High Abs Gap.3 Avg.40',
    'Low Abs Gap.3 Avg.40',
    'High Abs Gap.5 Avg.1',
    'Low Abs Gap.5 Avg.1',
    'High Abs Gap.5 Avg.3',
    'Low Abs Gap.5 Avg.3',
    'High Abs Gap.5 Avg.5',
    'Low Abs Gap.5 Avg.5',
    'High Abs Gap.5 Avg.10',
    'Low Abs Gap.5 Avg.10',
    'High Abs Gap.5 Avg.20',
    'Low Abs Gap.5 Avg.20',
    'High Abs Gap.5 Avg.40',
    'Low Abs Gap.5 Avg.40',
    'High Abs Gap.10 Avg.1',
    'Low Abs Gap.10 Avg.1',
    'High Abs Gap.10 Avg.3',
    'Low Abs Gap.10 Avg.3',
    'High Abs Gap.10 Avg.5',
    'Low Abs Gap.10 Avg.5',
    'High Abs Gap.10 Avg.10',
    'Low Abs Gap.10 Avg.10',
    'High Abs Gap.10 Avg.20',
    'Low Abs Gap.10 Avg.20',
    'High Abs Gap.10 Avg.40',
    'Low Abs Gap.10 Avg.40',
    'High Abs Gap.20 Avg.1',
    'Low Abs Gap.20 Avg.1',
    'High Abs Gap.20 Avg.3',
    'Low Abs Gap.20 Avg.3',
    'High Abs Gap.20 Avg.5',
    'Low Abs Gap.20 Avg.5',
    'High Abs Gap.20 Avg.10',
    'Low Abs Gap.20 Avg.10',
    'High Abs Gap.20 Avg.20',
    'Low Abs Gap.20 Avg.20',
    'High Abs Gap.20 Avg.40',
    'Low Abs Gap.20 Avg.40',
    'High Abs Gap.40 Avg.1',
    'Low Abs Gap.40 Avg.1',
    'High Abs Gap.40 Avg.3',
    'Low Abs Gap.40 Avg.3',
    'High Abs Gap.40 Avg.5',
    'Low Abs Gap.40 Avg.5',
    'High Abs Gap.40 Avg.10',
    'Low Abs Gap.40 Avg.10',
    'High Abs Gap.40 Avg.20',
    'Low Abs Gap.40 Avg.20',
    'High Abs Gap.40 Avg.40',
    'Low Abs Gap.40 Avg.40',
    'High Abs Gap Dev',
    'Low Abs Gap Dev',
    'High Abs Gap.1 Dev.1',
    'Low Abs Gap.1 Dev.1',
    'High Abs Gap.1 Dev.3',
    'Low Abs Gap.1 Dev.3',
    'High Abs Gap.1 Dev.5',
    'Low Abs Gap.1 Dev.5',
    'High Abs Gap.1 Dev.10',
    'Low Abs Gap.1 Dev.10',
    'High Abs Gap.1 Dev.20',
    'Low Abs Gap.1 Dev.20',
    'High Abs Gap.1 Dev.40',
    'Low Abs Gap.1 Dev.40',
    'High Abs Gap.3 Dev.1',
    'Low Abs Gap.3 Dev.1',
    'High Abs Gap.3 Dev.3',
    'Low Abs Gap.3 Dev.3',
    'High Abs Gap.3 Dev.5',
    'Low Abs Gap.3 Dev.5',
    'High Abs Gap.3 Dev.10',
    'Low Abs Gap.3 Dev.10',
    'High Abs Gap.3 Dev.20',
    'Low Abs Gap.3 Dev.20',
    'High Abs Gap.3 Dev.40',
    'Low Abs Gap.3 Dev.40',
    'High Abs Gap.5 Dev.1',
    'Low Abs Gap.5 Dev.1',
    'High Abs Gap.5 Dev.3',
    'Low Abs Gap.5 Dev.3',
    'High Abs Gap.5 Dev.5',
    'Low Abs Gap.5 Dev.5',
    'High Abs Gap.5 Dev.10',
    'Low Abs Gap.5 Dev.10',
    'High Abs Gap.5 Dev.20',
    'Low Abs Gap.5 Dev.20',
    'High Abs Gap.5 Dev.40',
    'Low Abs Gap.5 Dev.40',
    'High Abs Gap.10 Dev.1',
    'Low Abs Gap.10 Dev.1',
    'High Abs Gap.10 Dev.3',
    'Low Abs Gap.10 Dev.3',
    'High Abs Gap.10 Dev.5',
    'Low Abs Gap.10 Dev.5',
    'High Abs Gap.10 Dev.10',
    'Low Abs Gap.10 Dev.10',
    'High Abs Gap.10 Dev.20',
    'Low Abs Gap.10 Dev.20',
    'High Abs Gap.10 Dev.40',
    'Low Abs Gap.10 Dev.40',
    'High Abs Gap.20 Dev.1',
    'Low Abs Gap.20 Dev.1',
    'High Abs Gap.20 Dev.3',
    'Low Abs Gap.20 Dev.3',
    'High Abs Gap.20 Dev.5',
    'Low Abs Gap.20 Dev.5',
    'High Abs Gap.20 Dev.10',
    'Low Abs Gap.20 Dev.10',
    'High Abs Gap.20 Dev.20',
    'Low Abs Gap.20 Dev.20',
    'High Abs Gap.20 Dev.40',
    'Low Abs Gap.20 Dev.40',
    'High Abs Gap.40 Dev.1',
    'Low Abs Gap.40 Dev.1',
    'High Abs Gap.40 Dev.3',
    'Low Abs Gap.40 Dev.3',
    'High Abs Gap.40 Dev.5',
    'Low Abs Gap.40 Dev.5',
    'High Abs Gap.40 Dev.10',
    'Low Abs Gap.40 Dev.10',
    'High Abs Gap.40 Dev.20',
    'Low Abs Gap.40 Dev.20',
    'High Abs Gap.40 Dev.40',
    'Low Abs Gap.40 Dev.40',
    'High Abs Gap Lvl',
    'Low Abs Gap Lvl',
    'High Abs Gap.1 Lvl.1',
    'Low Abs Gap.1 Lvl.1',
    'High Abs Gap.1 Lvl.3',
    'Low Abs Gap.1 Lvl.3',
    'High Abs Gap.1 Lvl.5',
    'Low Abs Gap.1 Lvl.5',
    'High Abs Gap.1 Lvl.10',
    'Low Abs Gap.1 Lvl.10',
    'High Abs Gap.1 Lvl.20',
    'Low Abs Gap.1 Lvl.20',
    'High Abs Gap.1 Lvl.40',
    'Low Abs Gap.1 Lvl.40',
    'High Abs Gap.3 Lvl.1',
    'Low Abs Gap.3 Lvl.1',
    'High Abs Gap.3 Lvl.3',
    'Low Abs Gap.3 Lvl.3',
    'High Abs Gap.3 Lvl.5',
    'Low Abs Gap.3 Lvl.5',
    'High Abs Gap.3 Lvl.10',
    'Low Abs Gap.3 Lvl.10',
    'High Abs Gap.3 Lvl.20',
    'Low Abs Gap.3 Lvl.20',
    'High Abs Gap.3 Lvl.40',
    'Low Abs Gap.3 Lvl.40',
    'High Abs Gap.5 Lvl.1',
    'Low Abs Gap.5 Lvl.1',
    'High Abs Gap.5 Lvl.3',
    'Low Abs Gap.5 Lvl.3',
    'High Abs Gap.5 Lvl.5',
    'Low Abs Gap.5 Lvl.5',
    'High Abs Gap.5 Lvl.10',
    'Low Abs Gap.5 Lvl.10',
    'High Abs Gap.5 Lvl.20',
    'Low Abs Gap.5 Lvl.20',
    'High Abs Gap.5 Lvl.40',
    'Low Abs Gap.5 Lvl.40',
    'High Abs Gap.10 Lvl.1',
    'Low Abs Gap.10 Lvl.1',
    'High Abs Gap.10 Lvl.3',
    'Low Abs Gap.10 Lvl.3',
    'High Abs Gap.10 Lvl.5',
    'Low Abs Gap.10 Lvl.5',
    'High Abs Gap.10 Lvl.10',
    'Low Abs Gap.10 Lvl.10',
    'High Abs Gap.10 Lvl.20',
    'Low Abs Gap.10 Lvl.20',
    'High Abs Gap.10 Lvl.40',
    'Low Abs Gap.10 Lvl.40',
    'High Abs Gap.20 Lvl.1',
    'Low Abs Gap.20 Lvl.1',
    'High Abs Gap.20 Lvl.3',
    'Low Abs Gap.20 Lvl.3',
    'High Abs Gap.20 Lvl.5',
    'Low Abs Gap.20 Lvl.5',
    'High Abs Gap.20 Lvl.10',
    'Low Abs Gap.20 Lvl.10',
    'High Abs Gap.20 Lvl.20',
    'Low Abs Gap.20 Lvl.20',
    'High Abs Gap.20 Lvl.40',
    'Low Abs Gap.20 Lvl.40',
    'High Abs Gap.40 Lvl.1',
    'Low Abs Gap.40 Lvl.1',
    'High Abs Gap.40 Lvl.3',
    'Low Abs Gap.40 Lvl.3',
    'High Abs Gap.40 Lvl.5',
    'Low Abs Gap.40 Lvl.5',
    'High Abs Gap.40 Lvl.10',
    'Low Abs Gap.40 Lvl.10',
    'High Abs Gap.40 Lvl.20',
    'Low Abs Gap.40 Lvl.20',
    'High Abs Gap.40 Lvl.40',
    'Low Abs Gap.40 Lvl.40',
    'High Gap Var',
    'Low Gap Var',
    'High Gap.1 Var.1',
    'Low Gap.1 Var.1',
    'High Gap.1 Var.3',
    'Low Gap.1 Var.3',
    'High Gap.1 Var.5',
    'Low Gap.1 Var.5',
    'High Gap.1 Var.10',
    'Low Gap.1 Var.10',
    'High Gap.1 Var.20',
    'Low Gap.1 Var.20',
    'High Gap.1 Var.40',
    'Low Gap.1 Var.40',
    'High Gap.3 Var.1',
    'Low Gap.3 Var.1',
    'High Gap.3 Var.3',
    'Low Gap.3 Var.3',
    'High Gap.3 Var.5',
    'Low Gap.3 Var.5',
    'High Gap.3 Var.10',
    'Low Gap.3 Var.10',
    'High Gap.3 Var.20',
    'Low Gap.3 Var.20',
    'High Gap.3 Var.40',
    'Low Gap.3 Var.40',
    'High Gap.5 Var.1',
    'Low Gap.5 Var.1',
    'High Gap.5 Var.3',
    'Low Gap.5 Var.3',
    'High Gap.5 Var.5',
    'Low Gap.5 Var.5',
    'High Gap.5 Var.10',
    'Low Gap.5 Var.10',
    'High Gap.5 Var.20',
    'Low Gap.5 Var.20',
    'High Gap.5 Var.40',
    'Low Gap.5 Var.40',
    'High Gap.10 Var.1',
    'Low Gap.10 Var.1',
    'High Gap.10 Var.3',
    'Low Gap.10 Var.3',
    'High Gap.10 Var.5',
    'Low Gap.10 Var.5',
    'High Gap.10 Var.10',
    'Low Gap.10 Var.10',
    'High Gap.10 Var.20',
    'Low Gap.10 Var.20',
    'High Gap.10 Var.40',
    'Low Gap.10 Var.40',
    'High Gap.20 Var.1',
    'Low Gap.20 Var.1',
    'High Gap.20 Var.3',
    'Low Gap.20 Var.3',
    'High Gap.20 Var.5',
    'Low Gap.20 Var.5',
    'High Gap.20 Var.10',
    'Low Gap.20 Var.10',
    'High Gap.20 Var.20',
    'Low Gap.20 Var.20',
    'High Gap.20 Var.40',
    'Low Gap.20 Var.40',
    'High Gap.40 Var.1',
    'Low Gap.40 Var.1',
    'High Gap.40 Var.3',
    'Low Gap.40 Var.3',
    'High Gap.40 Var.5',
    'Low Gap.40 Var.5',
    'High Gap.40 Var.10',
    'Low Gap.40 Var.10',
    'High Gap.40 Var.20',
    'Low Gap.40 Var.20',
    'High Gap.40 Var.40',
    'Low Gap.40 Var.40'
]