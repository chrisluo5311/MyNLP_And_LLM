import random

if __name__ == '__main__':
    n = 5000
    pages = range(10000)
    row_total = random.randint(6500, 7000)
    key = random.sample(pages,row_total)
    print(len(key))

    # for each key, generate random number of distinct linked page
    open("report.txt", "w").close()
    with open("input.txt", "w") as f:
        for k in key:
            col_total = random.randint(1, n)
            values = random.sample(pages, col_total)
            f.write(f"{k}:{",".join(map(str,values))}\n")
