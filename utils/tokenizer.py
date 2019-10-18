import jieba


def segment_line(line):
    tokens = jieba.cut(line, cut_all=False)
    return " ".join(tokens)


def process_line(line):
    if isinstance(line, str):
        tokens = line.split("|")
        result = [segment_line(t) for t in tokens]
        return " | ".join(result)