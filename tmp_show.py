with open('app.py', encoding='utf-8') as f:
    for i, line in zip(range(1, 320), f):
        print(f"{i:03}: {line.rstrip()}")
