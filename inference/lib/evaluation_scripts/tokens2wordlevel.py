
def word_cleaner(x):
    return x.strip('\r\n')

def revert_from_sentence(sentence, subword_option):
    assert subword_option == None or subword_option == ""
    sentence = sentence.replace('Ġ', '')
    sentence = word_cleaner(sentence)
    return sentence

