from deeplearning.train import *

def identify_file(computer, file, lan_dic, lookup, max_len, vocab_size, bow, verbose=True):
    input=training_file_to_vocab_helper(file, lookup=lookup, max_len=max_len, vocab_size=vocab_size, bow=bow)
    input=torch.from_numpy(input).cuda().float().unsqueeze(0)
    y=computer(input).squeeze(0)
    pred_prob = F.softmax(y,dim=0)
    pred= torch.argmax(pred_prob).item()
    confidence=torch.max(pred_prob).item()
    for key, value in lan_dic.items():
        if value==pred:
            if verbose==True:
                print("The language is",key)
                print("Model confidence %.4f" % confidence)
                return key, confidence
            else:
                return key, confidence

def main_test(file):
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
    vocab_size=500
    bow=True
    max_len=100
    model= TransformerBOWMixed(vocab_size=vocab_size, d_model=256, d_inner=16, dropout=0.3, n_layers=8,
                               xi=0.5)
    optimizer=Adam(model.parameters(),lr=0.001)
    vocabuary = select_vocabulary(vocab_size - 2)
    train, valid = get_data_set(lan_dic)
    # turn vocabulary into a look-up dictionary
    lookup = {}
    for index, word in enumerate(vocabuary):
        lookup[word] = index
    for i, data_point in enumerate(train):
        if i > 10:
            break

    id_str = "mixedbow3"
    model, optimizer, highestepoch, highestiter = load_model(model, optimizer, 0, 0, id_str)

    return identify_file(model, file, lan_dic, lookup, max_len, vocab_size, bow)

if __name__=="__main__":
    dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
    file=Path(dir_path.parent/"testfile")
    main_test(file)