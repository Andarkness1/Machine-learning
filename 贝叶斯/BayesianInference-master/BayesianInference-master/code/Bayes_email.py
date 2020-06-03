# -*- coding: utf-8 -*-
# @Date     : 2017-05-09 09:29:13
# @Author   : Alan Lau (rlalan@outlook.com)
# @Language : Python3.5

# from fwalker import fun
# from reader import readtxt

import os

  # 按encoding方式按行读取path路径文件所有行，返回行列表lines
def readtxt(path,encoding):
    with open(path, 'r', encoding = encoding) as f:
        lines = f.readlines()
    return lines

 # 获取path路径下所有文件的绝对路径列表fileArray
def fileWalker(path):
    fileArray = []
    for root, dirs, files in os.walk(path):
        for fn in files:
            eachpath = str(root+'\\'+fn)
            fileArray.append(eachpath)
    return fileArray

# 去特殊字符标点符号，返回纯单词列表clean_word
def email_parser(email_path):
    punctuations = """,.<>()*&^%$#@!'";~`[]{}|、\\/~+_-=?"""
    content_list = readtxt(email_path, 'utf8')
    content = (' '.join(content_list)).replace('\r\n', ' ').replace('\t', ' ')
    clean_word = []
    for punctuation in punctuations:
        content = (' '.join(content.split(punctuation))).replace('  ', ' ')
        clean_word = [word.lower()
                      for word in content.split(' ') if len(word) > 2]
          # 此处去了停用词，可不去，影响不大
    return clean_word

 # 获取email_file路径下所有文件的总单词列表，append入word_list，extend入word_set并去重转为set
def get_word(email_file):
    word_list = []
    word_set = []
    punctuations = """,.<>()*&^%$#@!'";~`[]{}|、\\/~+_-=?"""
    email_paths = fileWalker(email_file)
    for email_path in email_paths:
        # content_list = readtxt(email_path, 'utf8')
        # content = (' '.join(content_list)).replace(
        #     '\r\n', ' ').replace('\t', ' ')
        # for punctuation in punctuations:
        #     # content = content.replace(punctuation, '').replace('  ', ' ')
        #     content = (' '.join(content.split(punctuation))).replace('  ', ' ')
        # clean_word = [word.lower()
        #               for word in content.split(' ') if len(word) > 2]
        clean_word = email_parser(email_path)
        word_list.append(clean_word)
        word_set.extend(clean_word)
    return word_list, set(word_set)

   # 返回训练集词频字典word_prob
def count_word_prob(email_list, union_set):
    word_prob = {}
    for word in union_set:
        counter = 0
        for email in email_list:
            if word in email:
                counter += 1
            else:
                continue
        prob = 0.0
        if counter != 0:
            prob = counter/len(email_list)
        else:
            prob = 0.01
        word_prob[word] = prob
    return word_prob

# 进行一次对测试集(10封邮件)的测试，输出对测试集的判断结果
    # 
def filter(ham_word_pro, spam_word_pro, test_file):
    test_paths = fileWalker(test_file)
    for test_path in test_paths:
        email_spam_prob = 0.0
        spam_prob = 0.5
        ham_prob = 0.5
        file_name = test_path.split('\\')[-1]
        prob_dict = {}
        words = set(email_parser(test_path))
        for word in words:
            Psw = 0.0
            if word not in spam_word_pro:
                Psw = 0.4
            else:
                Pws = spam_word_pro[word]
                Pwh = ham_word_pro[word]
                Psw = spam_prob*(Pws/(Pwh*ham_prob+Pws*spam_prob))
            prob_dict[word] = Psw
        numerator = 1
        denominator_h = 1
        for k, v in prob_dict.items():
            numerator *= v
            denominator_h *= (1-v)
        email_spam_prob = round(numerator/(numerator+denominator_h), 4)
        if email_spam_prob > 0.5:
            print(file_name, 'spam', email_spam_prob)
        else:
            print(file_name, 'ham', email_spam_prob)
        print(prob_dict)
        print('******************************************************')
        # break


def main():
    ham_file = r'..\email\ham'
    spam_file = r'..\email\spam'
    test_file = r'..\email\test'
    ham_list, ham_set = get_word(ham_file)
    spam_list, spam_set = get_word(spam_file)
    union_set = ham_set | spam_set
    ham_word_pro = count_word_prob(ham_list, union_set)
    spam_word_pro = count_word_prob(spam_list, union_set)
    # print(ham_set)
    filter(ham_word_pro, spam_word_pro, test_file)


if __name__ == '__main__':
    main()
