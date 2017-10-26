import os, sys
import numpy as np
from collections import Counter
from io import StringIO


def main():
  dir="./data/train-mails"
  wordDic=make_Dictionary(dir)
  wordCommonDic=removeStopWords(wordDic)

  print(wordCommonDic)

  features_matrix=extract_features(dir,wordCommonDic)
  np.save("features",features_matrix)
  fM=np.load("features.npy")

def removeStopWords(wordDic):
    list_to_remove = wordDic.keys()
    for item in list(list_to_remove):
        if item.isalpha() == False:
            del wordDic[item]
        elif len(item) == 1:
            del wordDic[item]

    wordCommonDic = wordDic.most_common(3000)
    return wordCommonDic

def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    all_words = []
    for mail in emails:
        with open(mail) as m:
            for i,line in enumerate(m):
                if i == 2:
                    words = line.split()
                    all_words += words

    dictionary = Counter(all_words)
    return dictionary

def extract_features(dir,wordDic):
    files = [os.path.join(dir, file) for file in os.listdir(dir)]
    features_matrix = np.zeros((len(files),3000))
    docID = 0;
    for file in files:
      with open(file) as fi:
        for i,line in enumerate(fi):
          if i == 2:
            words = line.split()
            for word in words:
              wordID = 0
              for i,d in enumerate(wordDic):
                if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = words.count(word)
        docID = docID + 1
    return features_matrix

if __name__ == '__main__':
    main()
