from lemmatizer import nkjp

if __name__ == '__main__':
    dict = nkjp.load_dict('data/dict/polimorf-20190818.tab')
    dict.print_check()
