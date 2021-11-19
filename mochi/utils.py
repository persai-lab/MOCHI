import os
import pathlib

prBold = lambda skk: print("\033[1m*{}\033[00m".format(skk))
prBlue = lambda skk: print("\033[34m*{}\033[00m".format(skk))
prRed = lambda skk: print("\033[91m*{}\033[00m".format(skk))
prGreen = lambda skk: print("\033[92m*{}\033[00m".format(skk))
prYellow = lambda skk: print("\033[93m*{}\033[00m".format(skk))
prLightPurple = lambda skk: print("\033[94m*{}\033[00m".format(skk))
prPurple = lambda skk: print("\033[95m*{}\033[00m".format(skk))
prCyan = lambda skk: print("\033[96m*{}\033[00m".format(skk))
prLightGray = lambda skk: print("\033[97m*{}\033[00m".format(skk))
prBlack = lambda skk: print("\033[98m*{}\033[00m".format(skk))

strBold = lambda skk: "\033[1m{}\033[00m".format(skk)
strBlue = lambda skk: "\033[34m{}\033[00m".format(skk)
strRed = lambda skk: "\033[91m{}\033[00m".format(skk)
strGreen = lambda skk: "\033[92m{}\033[00m".format(skk)
strYellow = lambda skk: "\033[93m{}\033[00m".format(skk)
strLightPurple = lambda skk: "\033[94m{}\033[00m".format(skk)
strPurple = lambda skk: "\033[95m{}\033[00m".format(skk)
strCyan = lambda skk: "\033[96m{}\033[00m".format(skk)
strLightGray = lambda skk: "\033[97m{}\033[00m".format(skk)
strBlack = lambda skk: "\033[98m{}\033[00m".format(skk)


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
