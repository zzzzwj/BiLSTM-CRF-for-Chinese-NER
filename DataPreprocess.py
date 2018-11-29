def sentence2words(mode, filename):
    with open(filename, 'rt') as f:
        w = open('data/{}_vec.txt'.format(mode), 'wt')
        lines = f.readlines()
        for i, line in enumerate(lines):
            print('\r{}/{}'.format(i + 1, len(lines)), end='')
            instances = line.rstrip().split(' ')
            for instance in instances:
                if mode == 'test':
                    w.write(instance + ' ' + 'O\n')
                else:
                    word, tag = instance.split('/')
                    w.write(word + ' ' + tag + '\n')
            w.write('\n')
        w.close()
        print('\nDone.')

def mergedata(filepath1, filepath2, output):
    final_lines = []
    with open(filepath1, 'rt') as f:
        final_lines += f.readlines()
    with open(filepath2, 'rt') as f:
        final_lines += f.readlines()
    with open(output, 'wt') as f:
        for line in final_lines:
            f.write(line)

if __name__ == '__main__':
    # mergedata('data/train_vec.txt', 'data/dev_vec.txt', 'data/merge_vec.txt')
    sentence2words('test', 'data/test.content.txt')