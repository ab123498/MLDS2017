import pickle
import string

def id2sent(dic, id_sents):
    inv_dic = {d[1]:d[0] for d in dic.items()}

    outputs = []
    for l in id_sents:
        l = [inv_dic[i] for i in l]

        cap = False
        s = string.capwords(l[0])
        for w in l[1:]:
            if w == '<eos>':
                break
            elif w == '<pad>':
                continue
            elif w == '.':
                cap = True
                s += w
            elif w in set(string.punctuation):
                s += w
            else:
                if cap:
                    w = string.capwords(w)
                    cap = False
                s += ' ' + w
        outputs.append(s)

    return outputs