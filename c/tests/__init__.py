"""c/tests is a regular package so that `python -m unittest tests.test_x`, the way
the Makefile and CI run modules from c/, imports these files even where
site-packages holds a `tests` package of its own (lightphe and ultralytics
install one): a directory without __init__.py is only a namespace portion, and a
regular package of the same name anywhere on sys.path wins over it.

`unittest discover -s tests` collects the same tests as before: its start
directory is the top level, and discover never imports a top level's __init__.py.
"""
