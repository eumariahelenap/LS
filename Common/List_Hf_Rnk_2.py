from typing import Literal as Lit


LI = [
    'High LS Rtn Rank',
    'High LS Rtn +1Hf',
    'High LS Rtn +2Hf',
    'High LS Rtn +4Hf',
    'High LS Rtn +8Hf',
    'High LS Rtn +16Hf',
    'High LS Rtn +32Hf',
    'High Cor Rank',
    'High Cor 1Hf',
    'High Cor 2Hf',
    'High Cor 4Hf',
    'High Cor 8Hf',
    'High Cor 16Hf',
    'High Cor 32Hf',
    'High Rank',
    'Low Rank',
    'High Gap',
    'Low Gap',
    'High Gap 1Hf',
    'Low Gap 1Hf',
    'High Gap 2Hf',
    'Low Gap 2Hf',
    'High Gap 4Hf',
    'Low Gap 4Hf',
    'High Gap 8Hf',
    'Low Gap 8Hf',
    'High Gap 16Hf',
    'Low Gap 16Hf',
    'High Gap 32Hf',
    'Low Gap 32Hf',
    'High Gap Avg',
    'Low Gap Avg',
    'High Gap.1 Avg.1',
    'Low Gap.1 Avg.1',
    'High Gap.1 Avg.2',
    'Low Gap.1 Avg.2',
    'High Gap.1 Avg.4',
    'Low Gap.1 Avg.4',
    'High Gap.1 Avg.8',
    'Low Gap.1 Avg.8',
    'High Gap.1 Avg.16',
    'Low Gap.1 Avg.16',
    'High Gap.1 Avg.32',
    'Low Gap.1 Avg.32',
    'High Gap.2 Avg.1',
    'Low Gap.2 Avg.1',
    'High Gap.2 Avg.2',
    'Low Gap.2 Avg.2',
    'High Gap.2 Avg.4',
    'Low Gap.2 Avg.4',
    'High Gap.2 Avg.8',
    'Low Gap.2 Avg.8',
    'High Gap.2 Avg.16',
    'Low Gap.2 Avg.16',
    'High Gap.2 Avg.32',
    'Low Gap.2 Avg.32',
    'High Gap.4 Avg.1',
    'Low Gap.4 Avg.1',
    'High Gap.4 Avg.2',
    'Low Gap.4 Avg.2',
    'High Gap.4 Avg.4',
    'Low Gap.4 Avg.4',
    'High Gap.4 Avg.8',
    'Low Gap.4 Avg.8',
    'High Gap.4 Avg.16',
    'Low Gap.4 Avg.16',
    'High Gap.4 Avg.32',
    'Low Gap.4 Avg.32',
    'High Gap.8 Avg.1',
    'Low Gap.8 Avg.1',
    'High Gap.8 Avg.2',
    'Low Gap.8 Avg.2',
    'High Gap.8 Avg.4',
    'Low Gap.8 Avg.4',
    'High Gap.8 Avg.8',
    'Low Gap.8 Avg.8',
    'High Gap.8 Avg.16',
    'Low Gap.8 Avg.16',
    'High Gap.8 Avg.32',
    'Low Gap.8 Avg.32',
    'High Gap.16 Avg.1',
    'Low Gap.16 Avg.1',
    'High Gap.16 Avg.2',
    'Low Gap.16 Avg.2',
    'High Gap.16 Avg.4',
    'Low Gap.16 Avg.4',
    'High Gap.16 Avg.8',
    'Low Gap.16 Avg.8',
    'High Gap.16 Avg.16',
    'Low Gap.16 Avg.16',
    'High Gap.16 Avg.32',
    'Low Gap.16 Avg.32',
    'High Gap.32 Avg.1',
    'Low Gap.32 Avg.1',
    'High Gap.32 Avg.2',
    'Low Gap.32 Avg.2',
    'High Gap.32 Avg.4',
    'Low Gap.32 Avg.4',
    'High Gap.32 Avg.8',
    'Low Gap.32 Avg.8',
    'High Gap.32 Avg.16',
    'Low Gap.32 Avg.16',
    'High Gap.32 Avg.32',
    'Low Gap.32 Avg.32',
    'High Gap Dev',
    'Low Gap Dev',
    'High Gap.1 Dev.1',
    'Low Gap.1 Dev.1',
    'High Gap.1 Dev.2',
    'Low Gap.1 Dev.2',
    'High Gap.1 Dev.4',
    'Low Gap.1 Dev.4',
    'High Gap.1 Dev.8',
    'Low Gap.1 Dev.8',
    'High Gap.1 Dev.16',
    'Low Gap.1 Dev.16',
    'High Gap.1 Dev.32',
    'Low Gap.1 Dev.32',
    'High Gap.2 Dev.1',
    'Low Gap.2 Dev.1',
    'High Gap.2 Dev.2',
    'Low Gap.2 Dev.2',
    'High Gap.2 Dev.4',
    'Low Gap.2 Dev.4',
    'High Gap.2 Dev.8',
    'Low Gap.2 Dev.8',
    'High Gap.2 Dev.16',
    'Low Gap.2 Dev.16',
    'High Gap.2 Dev.32',
    'Low Gap.2 Dev.32',
    'High Gap.4 Dev.1',
    'Low Gap.4 Dev.1',
    'High Gap.4 Dev.2',
    'Low Gap.4 Dev.2',
    'High Gap.4 Dev.4',
    'Low Gap.4 Dev.4',
    'High Gap.4 Dev.8',
    'Low Gap.4 Dev.8',
    'High Gap.4 Dev.16',
    'Low Gap.4 Dev.16',
    'High Gap.4 Dev.32',
    'Low Gap.4 Dev.32',
    'High Gap.8 Dev.1',
    'Low Gap.8 Dev.1',
    'High Gap.8 Dev.2',
    'Low Gap.8 Dev.2',
    'High Gap.8 Dev.4',
    'Low Gap.8 Dev.4',
    'High Gap.8 Dev.8',
    'Low Gap.8 Dev.8',
    'High Gap.8 Dev.16',
    'Low Gap.8 Dev.16',
    'High Gap.8 Dev.32',
    'Low Gap.8 Dev.32',
    'High Gap.16 Dev.1',
    'Low Gap.16 Dev.1',
    'High Gap.16 Dev.2',
    'Low Gap.16 Dev.2',
    'High Gap.16 Dev.4',
    'Low Gap.16 Dev.4',
    'High Gap.16 Dev.8',
    'Low Gap.16 Dev.8',
    'High Gap.16 Dev.16',
    'Low Gap.16 Dev.16',
    'High Gap.16 Dev.32',
    'Low Gap.16 Dev.32',
    'High Gap.32 Dev.1',
    'Low Gap.32 Dev.1',
    'High Gap.32 Dev.2',
    'Low Gap.32 Dev.2',
    'High Gap.32 Dev.4',
    'Low Gap.32 Dev.4',
    'High Gap.32 Dev.8',
    'Low Gap.32 Dev.8',
    'High Gap.32 Dev.16',
    'Low Gap.32 Dev.16',
    'High Gap.32 Dev.32',
    'Low Gap.32 Dev.32',
    'High Gap Lvl',
    'Low Gap Lvl',
    'High Gap.1 Lvl.1',
    'Low Gap.1 Lvl.1',
    'High Gap.1 Lvl.2',
    'Low Gap.1 Lvl.2',
    'High Gap.1 Lvl.4',
    'Low Gap.1 Lvl.4',
    'High Gap.1 Lvl.8',
    'Low Gap.1 Lvl.8',
    'High Gap.1 Lvl.16',
    'Low Gap.1 Lvl.16',
    'High Gap.1 Lvl.32',
    'Low Gap.1 Lvl.32',
    'High Gap.2 Lvl.1',
    'Low Gap.2 Lvl.1',
    'High Gap.2 Lvl.2',
    'Low Gap.2 Lvl.2',
    'High Gap.2 Lvl.4',
    'Low Gap.2 Lvl.4',
    'High Gap.2 Lvl.8',
    'Low Gap.2 Lvl.8',
    'High Gap.2 Lvl.16',
    'Low Gap.2 Lvl.16',
    'High Gap.2 Lvl.32',
    'Low Gap.2 Lvl.32',
    'High Gap.4 Lvl.1',
    'Low Gap.4 Lvl.1',
    'High Gap.4 Lvl.2',
    'Low Gap.4 Lvl.2',
    'High Gap.4 Lvl.4',
    'Low Gap.4 Lvl.4',
    'High Gap.4 Lvl.8',
    'Low Gap.4 Lvl.8',
    'High Gap.4 Lvl.16',
    'Low Gap.4 Lvl.16',
    'High Gap.4 Lvl.32',
    'Low Gap.4 Lvl.32',
    'High Gap.8 Lvl.1',
    'Low Gap.8 Lvl.1',
    'High Gap.8 Lvl.2',
    'Low Gap.8 Lvl.2',
    'High Gap.8 Lvl.4',
    'Low Gap.8 Lvl.4',
    'High Gap.8 Lvl.8',
    'Low Gap.8 Lvl.8',
    'High Gap.8 Lvl.16',
    'Low Gap.8 Lvl.16',
    'High Gap.8 Lvl.32',
    'Low Gap.8 Lvl.32',
    'High Gap.16 Lvl.1',
    'Low Gap.16 Lvl.1',
    'High Gap.16 Lvl.2',
    'Low Gap.16 Lvl.2',
    'High Gap.16 Lvl.4',
    'Low Gap.16 Lvl.4',
    'High Gap.16 Lvl.8',
    'Low Gap.16 Lvl.8',
    'High Gap.16 Lvl.16',
    'Low Gap.16 Lvl.16',
    'High Gap.16 Lvl.32',
    'Low Gap.16 Lvl.32',
    'High Gap.32 Lvl.1',
    'Low Gap.32 Lvl.1',
    'High Gap.32 Lvl.2',
    'Low Gap.32 Lvl.2',
    'High Gap.32 Lvl.4',
    'Low Gap.32 Lvl.4',
    'High Gap.32 Lvl.8',
    'Low Gap.32 Lvl.8',
    'High Gap.32 Lvl.16',
    'Low Gap.32 Lvl.16',
    'High Gap.32 Lvl.32',
    'Low Gap.32 Lvl.32',
    'High Abs Gap',
    'Low Abs Gap',
    'High Abs Gap 1Hf',
    'Low Abs Gap 1Hf',
    'High Abs Gap 2Hf',
    'Low Abs Gap 2Hf',
    'High Abs Gap 4Hf',
    'Low Abs Gap 4Hf',
    'High Abs Gap 8Hf',
    'Low Abs Gap 8Hf',
    'High Abs Gap 16Hf',
    'Low Abs Gap 16Hf',
    'High Abs Gap 32Hf',
    'Low Abs Gap 32Hf',
    'High Abs Gap Avg',
    'Low Abs Gap Avg',
    'High Abs Gap.1 Avg.1',
    'Low Abs Gap.1 Avg.1',
    'High Abs Gap.1 Avg.2',
    'Low Abs Gap.1 Avg.2',
    'High Abs Gap.1 Avg.4',
    'Low Abs Gap.1 Avg.4',
    'High Abs Gap.1 Avg.8',
    'Low Abs Gap.1 Avg.8',
    'High Abs Gap.1 Avg.16',
    'Low Abs Gap.1 Avg.16',
    'High Abs Gap.1 Avg.32',
    'Low Abs Gap.1 Avg.32',
    'High Abs Gap.2 Avg.1',
    'Low Abs Gap.2 Avg.1',
    'High Abs Gap.2 Avg.2',
    'Low Abs Gap.2 Avg.2',
    'High Abs Gap.2 Avg.4',
    'Low Abs Gap.2 Avg.4',
    'High Abs Gap.2 Avg.8',
    'Low Abs Gap.2 Avg.8',
    'High Abs Gap.2 Avg.16',
    'Low Abs Gap.2 Avg.16',
    'High Abs Gap.2 Avg.32',
    'Low Abs Gap.2 Avg.32',
    'High Abs Gap.4 Avg.1',
    'Low Abs Gap.4 Avg.1',
    'High Abs Gap.4 Avg.2',
    'Low Abs Gap.4 Avg.2',
    'High Abs Gap.4 Avg.4',
    'Low Abs Gap.4 Avg.4',
    'High Abs Gap.4 Avg.8',
    'Low Abs Gap.4 Avg.8',
    'High Abs Gap.4 Avg.16',
    'Low Abs Gap.4 Avg.16',
    'High Abs Gap.4 Avg.32',
    'Low Abs Gap.4 Avg.32',
    'High Abs Gap.8 Avg.1',
    'Low Abs Gap.8 Avg.1',
    'High Abs Gap.8 Avg.2',
    'Low Abs Gap.8 Avg.2',
    'High Abs Gap.8 Avg.4',
    'Low Abs Gap.8 Avg.4',
    'High Abs Gap.8 Avg.8',
    'Low Abs Gap.8 Avg.8',
    'High Abs Gap.8 Avg.16',
    'Low Abs Gap.8 Avg.16',
    'High Abs Gap.8 Avg.32',
    'Low Abs Gap.8 Avg.32',
    'High Abs Gap.16 Avg.1',
    'Low Abs Gap.16 Avg.1',
    'High Abs Gap.16 Avg.2',
    'Low Abs Gap.16 Avg.2',
    'High Abs Gap.16 Avg.4',
    'Low Abs Gap.16 Avg.4',
    'High Abs Gap.16 Avg.8',
    'Low Abs Gap.16 Avg.8',
    'High Abs Gap.16 Avg.16',
    'Low Abs Gap.16 Avg.16',
    'High Abs Gap.16 Avg.32',
    'Low Abs Gap.16 Avg.32',
    'High Abs Gap.32 Avg.1',
    'Low Abs Gap.32 Avg.1',
    'High Abs Gap.32 Avg.2',
    'Low Abs Gap.32 Avg.2',
    'High Abs Gap.32 Avg.4',
    'Low Abs Gap.32 Avg.4',
    'High Abs Gap.32 Avg.8',
    'Low Abs Gap.32 Avg.8',
    'High Abs Gap.32 Avg.16',
    'Low Abs Gap.32 Avg.16',
    'High Abs Gap.32 Avg.32',
    'Low Abs Gap.32 Avg.32',
    'High Abs Gap Dev',
    'Low Abs Gap Dev',
    'High Abs Gap.1 Dev.1',
    'Low Abs Gap.1 Dev.1',
    'High Abs Gap.1 Dev.2',
    'Low Abs Gap.1 Dev.2',
    'High Abs Gap.1 Dev.4',
    'Low Abs Gap.1 Dev.4',
    'High Abs Gap.1 Dev.8',
    'Low Abs Gap.1 Dev.8',
    'High Abs Gap.1 Dev.16',
    'Low Abs Gap.1 Dev.16',
    'High Abs Gap.1 Dev.32',
    'Low Abs Gap.1 Dev.32',
    'High Abs Gap.2 Dev.1',
    'Low Abs Gap.2 Dev.1',
    'High Abs Gap.2 Dev.2',
    'Low Abs Gap.2 Dev.2',
    'High Abs Gap.2 Dev.4',
    'Low Abs Gap.2 Dev.4',
    'High Abs Gap.2 Dev.8',
    'Low Abs Gap.2 Dev.8',
    'High Abs Gap.2 Dev.16',
    'Low Abs Gap.2 Dev.16',
    'High Abs Gap.2 Dev.32',
    'Low Abs Gap.2 Dev.32',
    'High Abs Gap.4 Dev.1',
    'Low Abs Gap.4 Dev.1',
    'High Abs Gap.4 Dev.2',
    'Low Abs Gap.4 Dev.2',
    'High Abs Gap.4 Dev.4',
    'Low Abs Gap.4 Dev.4',
    'High Abs Gap.4 Dev.8',
    'Low Abs Gap.4 Dev.8',
    'High Abs Gap.4 Dev.16',
    'Low Abs Gap.4 Dev.16',
    'High Abs Gap.4 Dev.32',
    'Low Abs Gap.4 Dev.32',
    'High Abs Gap.8 Dev.1',
    'Low Abs Gap.8 Dev.1',
    'High Abs Gap.8 Dev.2',
    'Low Abs Gap.8 Dev.2',
    'High Abs Gap.8 Dev.4',
    'Low Abs Gap.8 Dev.4',
    'High Abs Gap.8 Dev.8',
    'Low Abs Gap.8 Dev.8',
    'High Abs Gap.8 Dev.16',
    'Low Abs Gap.8 Dev.16',
    'High Abs Gap.8 Dev.32',
    'Low Abs Gap.8 Dev.32',
    'High Abs Gap.16 Dev.1',
    'Low Abs Gap.16 Dev.1',
    'High Abs Gap.16 Dev.2',
    'Low Abs Gap.16 Dev.2',
    'High Abs Gap.16 Dev.4',
    'Low Abs Gap.16 Dev.4',
    'High Abs Gap.16 Dev.8',
    'Low Abs Gap.16 Dev.8',
    'High Abs Gap.16 Dev.16',
    'Low Abs Gap.16 Dev.16',
    'High Abs Gap.16 Dev.32',
    'Low Abs Gap.16 Dev.32',
    'High Abs Gap.32 Dev.1',
    'Low Abs Gap.32 Dev.1',
    'High Abs Gap.32 Dev.2',
    'Low Abs Gap.32 Dev.2',
    'High Abs Gap.32 Dev.4',
    'Low Abs Gap.32 Dev.4',
    'High Abs Gap.32 Dev.8',
    'Low Abs Gap.32 Dev.8',
    'High Abs Gap.32 Dev.16',
    'Low Abs Gap.32 Dev.16',
    'High Abs Gap.32 Dev.32',
    'Low Abs Gap.32 Dev.32',
    'High Abs Gap Lvl',
    'Low Abs Gap Lvl',
    'High Abs Gap.1 Lvl.1',
    'Low Abs Gap.1 Lvl.1',
    'High Abs Gap.1 Lvl.2',
    'Low Abs Gap.1 Lvl.2',
    'High Abs Gap.1 Lvl.4',
    'Low Abs Gap.1 Lvl.4',
    'High Abs Gap.1 Lvl.8',
    'Low Abs Gap.1 Lvl.8',
    'High Abs Gap.1 Lvl.16',
    'Low Abs Gap.1 Lvl.16',
    'High Abs Gap.1 Lvl.32',
    'Low Abs Gap.1 Lvl.32',
    'High Abs Gap.2 Lvl.1',
    'Low Abs Gap.2 Lvl.1',
    'High Abs Gap.2 Lvl.2',
    'Low Abs Gap.2 Lvl.2',
    'High Abs Gap.2 Lvl.4',
    'Low Abs Gap.2 Lvl.4',
    'High Abs Gap.2 Lvl.8',
    'Low Abs Gap.2 Lvl.8',
    'High Abs Gap.2 Lvl.16',
    'Low Abs Gap.2 Lvl.16',
    'High Abs Gap.2 Lvl.32',
    'Low Abs Gap.2 Lvl.32',
    'High Abs Gap.4 Lvl.1',
    'Low Abs Gap.4 Lvl.1',
    'High Abs Gap.4 Lvl.2',
    'Low Abs Gap.4 Lvl.2',
    'High Abs Gap.4 Lvl.4',
    'Low Abs Gap.4 Lvl.4',
    'High Abs Gap.4 Lvl.8',
    'Low Abs Gap.4 Lvl.8',
    'High Abs Gap.4 Lvl.16',
    'Low Abs Gap.4 Lvl.16',
    'High Abs Gap.4 Lvl.32',
    'Low Abs Gap.4 Lvl.32',
    'High Abs Gap.8 Lvl.1',
    'Low Abs Gap.8 Lvl.1',
    'High Abs Gap.8 Lvl.2',
    'Low Abs Gap.8 Lvl.2',
    'High Abs Gap.8 Lvl.4',
    'Low Abs Gap.8 Lvl.4',
    'High Abs Gap.8 Lvl.8',
    'Low Abs Gap.8 Lvl.8',
    'High Abs Gap.8 Lvl.16',
    'Low Abs Gap.8 Lvl.16',
    'High Abs Gap.8 Lvl.32',
    'Low Abs Gap.8 Lvl.32',
    'High Abs Gap.16 Lvl.1',
    'Low Abs Gap.16 Lvl.1',
    'High Abs Gap.16 Lvl.2',
    'Low Abs Gap.16 Lvl.2',
    'High Abs Gap.16 Lvl.4',
    'Low Abs Gap.16 Lvl.4',
    'High Abs Gap.16 Lvl.8',
    'Low Abs Gap.16 Lvl.8',
    'High Abs Gap.16 Lvl.16',
    'Low Abs Gap.16 Lvl.16',
    'High Abs Gap.16 Lvl.32',
    'Low Abs Gap.16 Lvl.32',
    'High Abs Gap.32 Lvl.1',
    'Low Abs Gap.32 Lvl.1',
    'High Abs Gap.32 Lvl.2',
    'Low Abs Gap.32 Lvl.2',
    'High Abs Gap.32 Lvl.4',
    'Low Abs Gap.32 Lvl.4',
    'High Abs Gap.32 Lvl.8',
    'Low Abs Gap.32 Lvl.8',
    'High Abs Gap.32 Lvl.16',
    'Low Abs Gap.32 Lvl.16',
    'High Abs Gap.32 Lvl.32',
    'Low Abs Gap.32 Lvl.32',
    'High Gap Var',
    'Low Gap Var',
    'High Gap.1 Var.1',
    'Low Gap.1 Var.1',
    'High Gap.1 Var.2',
    'Low Gap.1 Var.2',
    'High Gap.1 Var.4',
    'Low Gap.1 Var.4',
    'High Gap.1 Var.8',
    'Low Gap.1 Var.8',
    'High Gap.1 Var.16',
    'Low Gap.1 Var.16',
    'High Gap.1 Var.32',
    'Low Gap.1 Var.32',
    'High Gap.2 Var.1',
    'Low Gap.2 Var.1',
    'High Gap.2 Var.2',
    'Low Gap.2 Var.2',
    'High Gap.2 Var.4',
    'Low Gap.2 Var.4',
    'High Gap.2 Var.8',
    'Low Gap.2 Var.8',
    'High Gap.2 Var.16',
    'Low Gap.2 Var.16',
    'High Gap.2 Var.32',
    'Low Gap.2 Var.32',
    'High Gap.4 Var.1',
    'Low Gap.4 Var.1',
    'High Gap.4 Var.2',
    'Low Gap.4 Var.2',
    'High Gap.4 Var.4',
    'Low Gap.4 Var.4',
    'High Gap.4 Var.8',
    'Low Gap.4 Var.8',
    'High Gap.4 Var.16',
    'Low Gap.4 Var.16',
    'High Gap.4 Var.32',
    'Low Gap.4 Var.32',
    'High Gap.8 Var.1',
    'Low Gap.8 Var.1',
    'High Gap.8 Var.2',
    'Low Gap.8 Var.2',
    'High Gap.8 Var.4',
    'Low Gap.8 Var.4',
    'High Gap.8 Var.8',
    'Low Gap.8 Var.8',
    'High Gap.8 Var.16',
    'Low Gap.8 Var.16',
    'High Gap.8 Var.32',
    'Low Gap.8 Var.32',
    'High Gap.16 Var.1',
    'Low Gap.16 Var.1',
    'High Gap.16 Var.2',
    'Low Gap.16 Var.2',
    'High Gap.16 Var.4',
    'Low Gap.16 Var.4',
    'High Gap.16 Var.8',
    'Low Gap.16 Var.8',
    'High Gap.16 Var.16',
    'Low Gap.16 Var.16',
    'High Gap.16 Var.32',
    'Low Gap.16 Var.32',
    'High Gap.32 Var.1',
    'Low Gap.32 Var.1',
    'High Gap.32 Var.2',
    'Low Gap.32 Var.2',
    'High Gap.32 Var.4',
    'Low Gap.32 Var.4',
    'High Gap.32 Var.8',
    'Low Gap.32 Var.8',
    'High Gap.32 Var.16',
    'Low Gap.32 Var.16',
    'High Gap.32 Var.32',
    'Low Gap.32 Var.32'
]