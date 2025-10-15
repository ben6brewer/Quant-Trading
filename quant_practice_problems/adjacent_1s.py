def has_adjacent_1s(n):
    number = str(n)
    for i in range(len(number) - 1):
        if number[i] == '1' and number[i+1] == '1':
            return True
    return False

top_of_range = 1000

counter = 0
for i in range(top_of_range):
    if not has_adjacent_1s(i):
        counter += 1
print(f"valid numbers in range {top_of_range}: {counter} ({round((counter / top_of_range),2)}%)")


