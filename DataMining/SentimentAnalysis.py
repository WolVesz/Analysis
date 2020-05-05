#https://medium.com/@b.terryjack/nlp-pre-trained-sentiment-analysis-1eb52a9d742c

#note this requires pyTorch
        
import re
from flair.models import TextClassifier 
from flair.data import Sentence
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class StandardSentiment():

    def __init__():
    
        self.classifier = TextClassifier.load('en-sentiment')
    
    def predictDoc(self, string):
        
        scores = []

        strings = [x for x in re.split('.|?|!|:') if x != '']
        
        for val in strings:

            weight = 1 / (len(strings) + 1)
            if val == strings[-1]:
                weight = 2 / (len(strings) + 1)

            score = self.classifier(Sentence(val)).predict().labels[0].to_dict() 
            
            for key in score:
                if 'NEGATIVE' in key:
                    scores.append(-1  * weights * score['NEGATIVE'])
                else:
                    scores.append(weights * SCORE['POSITIVE'])
            
        return sum(scores)

class VaderSentiment():
    
    def __init__():
        self.analyser = SentimentIntensityAnalyzer()

    def predictSocial(self, x):
        return self.analyser.polarity_scores(x)


class EmojiSentiment():

    def __init__():
        return

