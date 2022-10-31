infile = open('file_test.txt')

header = infile.read(3)

print(header)

header = infile.read(4)

print(header)

infile.seek(4)

print(infile.read(3))


