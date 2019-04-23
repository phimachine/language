# some modules to compress the text space for n-gram based approach
# we need to compute the most frequent characters from the files

from pathlib import Path
corpus_dir=Path("D:")/"Git"/"trdata"/"languages"

def possible_chars():
    lan_dic = {"C": 0,
               "C#": 1,
               "C++": 2,
               "Go": 3,
               "Java": 4,
               "Javascript": 5,
               "Lua": 6,
               "Objective-C": 7,
               "Python": 8,
               "Ruby": 9,
               "Rust": 10,
               "Shell": 11}
    
    charset=set()

    for lang in lan_dic:
        dir=corpus_dir/lang
        for idx,file in enumerate(dir.iterdir()):
            if idx>100:
                break

            with open(file, 'r', encoding="utf8", errors='ignore') as file:
                while True:
                    c = file.read(1)
                    if not c:
                        break
                    c=c.encode('ascii',errors='ignore').decode('ascii')
                    charset.add(c)

    print(charset)
    # this is almost the same to string.printable
    return charset


if __name__ == '__main__':
    possible_chars()