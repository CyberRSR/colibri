#!/usr/bin/env python3
"""make_requests.py <folder>: the long answer's request -- a whole Python module asked for in a short prompt,
code only, greedy (temperature 0), 2,048 tokens -- as <folder>/code.json."""
import json
import os
import sys

NGEN = 2048

CODE = (
    "Write a complete, self-contained Python 3 module named minisql.py that implements a small SQL engine "
    "over CSV files. It needs: a tokenizer; a recursive-descent parser for SELECT (a column list, *, "
    "expressions, aliases with AS), FROM (one CSV file), WHERE (AND, OR, NOT, parentheses, the comparisons "
    "=, !=, <, <=, >, >=, LIKE, IN, BETWEEN, IS NULL), GROUP BY with COUNT, SUM, AVG, MIN and MAX, HAVING, "
    "ORDER BY (several keys, ASC and DESC) and LIMIT with OFFSET; an evaluator; type inference for numbers "
    "in the CSV cells; clear error messages with the position of the error; a command-line interface with "
    "an interactive REPL; a docstring for every class and function; and a thorough unittest suite at the "
    "end of the file. Write the whole file. No explanations before or after the code.")


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    body = {"model": "kimi-k3.1-pro", "messages": [{"role": "user", "content": CODE}],
            "max_tokens": NGEN, "temperature": 0}
    path = os.path.join(out, "code.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(body, f, ensure_ascii=False)
    print(path)


if __name__ == "__main__":
    main()
