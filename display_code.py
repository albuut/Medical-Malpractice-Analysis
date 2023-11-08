import sys

file_name = sys.argv[1]
from_line = int(sys.argv[2])
to_line = int(sys.argv[3])

with open(file_name, 'r') as file:
    print(r'``` python')
    lines = file.readlines()[from_line-1:to_line]
    for line in lines:
        print(line)
    print(r'```')