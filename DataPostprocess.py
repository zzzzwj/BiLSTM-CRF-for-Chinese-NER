def mergetag(tag_list, nexttag):
    if tag_list[0] == 'O':
        return 'O'
    if tag_list[0].startswith('B-') and tag_list[len(tag_list) - 1].startswith('I-'):
        return tag_list[0]
    elif tag_list[0].startswith('B-'):
        return tag_list[0]
    elif tag_list[len(tag_list) - 1].startswith('I-'):
        if nexttag == 'O' or nexttag is None:
            return 'O' + tag_list[len(tag_list) - 1][1:]
        else:
            return 'I' + tag_list[len(tag_list) - 1][1:]
    else:
        return 'O'

if __name__ == '__main__':
    sentence_list = []
    with open('result.txt', 'rt') as f:
        lines = f.readlines()
        sentence = []
        for line in lines:
            if line != '\n':
                word, tag = line.rstrip().split(' ')
                sentence.append((word, tag))
            else:
                sentence_list.append(sentence.copy())
                sentence.clear()
    # print(sentence_list[0])

    with open('test.content.txt', 'rt') as f:
        tags_list = []
        lines = f.readlines()
        for i, line in enumerate(lines):
            words_list = line.rstrip().split()
            sentence = sentence_list[i]
            tags = []
            idx = 0
            for words in words_list:
                buf = ''
                tag = []
                while len(buf) <= len(words):
                    buf += sentence[idx][0]
                    tag.append(sentence[idx][1])
                    idx += 1
                    if words == buf:
                        break
                tags.append(mergetag(tag, sentence[idx + 1][1] if idx + 1 < len(sentence) else None))
            tags_list.append(tags)
    with open('final_result.txt', 'wt') as f:
        for tags in tags_list:
            for i in range(len(tags) - 1):
                f.write(tags[i] + ' ')
            f.write(tags[len(tags) - 1] + '\n')
    print('Done.')