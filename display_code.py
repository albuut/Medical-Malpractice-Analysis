import sys

file_name = sys.argv[1]
from_line = sys.argv[2]
to_line = sys.argv[3]

with open(file_name, 'r') as file:
    lines = file.readlines()[from_line-1:to_line]
    for line in lines:
        print(line)