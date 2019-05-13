import sys

fname = sys.argv[1]
with open(fname, 'r') as f:
  lines = f.readlines()
  lines = [line.strip() for line in lines]

  if len(lines) < 10000:
    raise ValueError('Invalid number of predictions')

  lines_unique = set(lines)
  if lines_unique != set(['0', '1']):
    raise ValueError('Invalid predictions')

print('Valid predictions file')
