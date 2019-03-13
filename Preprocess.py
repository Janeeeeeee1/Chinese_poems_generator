import collections

def get_poems(poem_file="./poetry.txt"):
    poems = []
    with open(poem_file, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            try:
                title, content = line.strip().split(":")
                content = content.replace(" ", "")
                if '_' in content or '(' in content or \
                '（' in content or '《' in content or \
                '[' in content :
                    continue
                if len(content) < 5 or len(content) > 79:
                    continue
                # 设置初始和结束标志
                content = '[' + content + ']'
                poems.append(content)
            except Exception as e:
                pass
        return poems

def build_dataset():
    poems = get_poems()
    # 按照诗的长度排列
    poems = sorted(poems, key=lambda line: len(line))
    print("唐诗总数:", len(poems))
    words = []
    # 把诗中的每一个都放到一个列表中（可重复）
    for poetry in poems:
        words += [word for word in poetry]
    # 计算每个字的频率
    counter = collections.Counter(words)
    # 从大到小排序
    counter_sorted = sorted(counter.items(), key=lambda x: -x[1])
    # 从counter中解压，并获取当中的词
    words, _ = zip(*counter_sorted)
    words += (" ", )
    # word -> id
    word2int = dict(zip(words, range(len(words))))
    # id -> word
    int2word = dict(zip(word2int.values(), word2int.keys()))

    poem_vectors = [[word2int[word] for word in poem] for poem in poems]
    return word2int, poem_vectors, int2word

if __name__ == '__main__':
    word2int,poem_vectors,int2word = build_dataset()
    print(word2int)
    print('\n')
    print(poem_vectors)
    print('\n')
    print(int2word)