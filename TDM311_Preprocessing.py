import string #string contains helper function with list of punctuation

class Stopwordremover:
    def __init__(self, *params):
        with open("TDM311_Stopwords.txt", 'r') as fp: #r is readmode
          self.stopwords_list = [line.strip() for line in fp.readlines()]

    def preprocess(self, texts):
        """Implements text preprocessing

        :param texts: text lines
        :type texts:  list

        :return: preprocessed lines
        :rtype:  list[str]
        """
        # Create a working copy with stripped lowercsae text
        out_texts = [text.strip().lower() for text in texts]
        
        # Remove punctuation
        # First split into individual words
        words = [text.split() for text in out_texts]
        # Then remove each punctuation from words and strip.. TERRIBLE list comprehension but oh well
        words = [[''.join([c if c not in string.punctuation else ' ' for c in word]).strip() for word in word_list] for word_list in words]
        out_texts = [' '.join(word_list) for word_list in words]
        
        # Remove all words in the stop word list
        for i, text in enumerate(out_texts):
            words = text.split(' ')
            words = [word for word in words if word not in self.stopwords_list]
            out_texts[i] = ' '.join(words)
        
        return out_texts
    
stopwordremover = Stopwordremover()
import pandas
documents = pandas.read_csv("labeled_text_data.csv")
documents["transcription"] = stopwordremover.preprocess(documents["transcription"]) 
documents.to_csv("TDM311_Preprocessing.csv")