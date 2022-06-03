# import os

# from CP.Base import Base

from os.path import dirname, basename, isfile, join
import glob
modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

# for filename in os.listdir("D:/DTU/project-work-DTU-conformal-prediction/CP"):
#     if filename[:2] != '__' and filename[-2:] == '.py':
#         with open(filename) as file:
#             from file import *
