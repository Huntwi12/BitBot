with open('app.py', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        if i > 600:
            break
        print(f"{i:03}: {line.rstrip()}")
