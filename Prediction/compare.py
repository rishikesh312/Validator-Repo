import difflib
with open('predictions.txt') as f1:
    f1_text = f1.read()
with open('Tappredict.txt') as f2:
    f2_text = f2.read()
# Find and print the diff:
for line in difflib.unified_diff(f1_text, f2_text, fromfile='file1', tofile='file2', lineterm=''):
    print(line)