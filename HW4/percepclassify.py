import json
import re
import sys


class PerceptronClassify:
    def __init__(self, model_file, test_file):
        self.ids = []
        self.reviews = []

        self.test_words = dict()
        self.predict_true_fake = []
        self.predict_pos_neg = []

        self.read_model(model_file)
        self.read_test(test_file)


    def read_model(self, filename):
        """
        Read the model file and initialize the weights and bias
        """
        with open(filename, 'r') as f:
            json_data = json.load(f)
        self.data = json_data
        self.vanilla_weights1 = json_data["weights1"]
        self.vanilla_weights2 = json_data["weights2"]
        self.vanilla_bias1 = json_data["bias1"]
        self.vanilla_bias2 = json_data["bias2"]
        self.stopwords = json_data["stopwords"]
        self.words = json_data["words"]


    def read_test(self,filename):
        """
        Read the test file and store the ids and reviews
        """
        with open(filename, 'r') as f:
            lines = f.read().splitlines()

        for row in lines:
            row_data = row.split(" ")
            self.ids.append(row_data[0])
            self.reviews.append(" ".join(row_data[i].lower() for i in range(1,len(row_data))))

        self.remove_punctuation()
        self.remove_stopwords()
        
        self.word_count()
    

    def remove_stopwords(self):
        """
        Remove the stopwords from the reviews
        """
        for i in range(len(self.reviews)):
            self.reviews[i] = " ".join([w for w in self.reviews[i].split() if w not in self.stopwords])

    
    def remove_punctuation(self):
        """
        Remove the punctuation from the reviews using regular expressions
        """
        # remove punctuation using regex
        for i in range(len(self.reviews)):
            self.reviews[i] = re.sub(r'[^\w\s]','',self.reviews[i])


    def word_count(self):
        """
        Count the number of words in the reviews
        """
        for review in self.reviews:
            for word in review.split():
                if word=='': continue
                if word in self.words and word not in self.stopwords:
                    if word not in self.test_words:
                        self.test_words[word] = 1
                    else:
                        self.test_words[word] += 1


    def classify(self):
        """
        Classify the reviews as positive or negative and true or fake
        """
        for review in self.reviews:
            test_words = dict()
            for word in review.split():
                if word=='': continue
                if word in self.words and word not in self.stopwords:
                    if word not in test_words:
                        test_words[word] = 1
                    else:
                        test_words[word] += 1
                

            activation1 = 0
            activation2 = 0

            for word in test_words:
                if word not in self.vanilla_weights1:
                    self.vanilla_weights1[word] = 0
                if word not in self.vanilla_weights2:
                    self.vanilla_weights2[word] = 0

                #only update the weights that are in the training data and ignore otherwise
                if word in self.words:
                    activation1 += self.vanilla_weights1[word] * test_words[word]
                    activation2 += self.vanilla_weights2[word] * test_words[word]
                    
            activation1 += self.vanilla_bias1
            activation2 += self.vanilla_bias2

            if activation1 > 0:
                self.predict_true_fake.append("True")
            else:
                self.predict_true_fake.append("Fake")
            
            if activation2 > 0:
                self.predict_pos_neg.append("Pos")
            else:
                self.predict_pos_neg.append("Neg")


    def write_predictions(self, filename):
        """
        Write the predictions to the output file
        """
        with open(filename, 'w') as f:
            for i in range(len(self.ids)):
                f.write(self.ids[i] + " " + self.predict_true_fake[i] + " " + self.predict_pos_neg[i] + "\n")



if __name__ == "__main__":
    model_file = sys.argv[1]
    test_file = sys.argv[2]
    
    percepClassify = PerceptronClassify(model_file, test_file)
    percepClassify.classify()
    percepClassify.write_predictions("percepoutput.txt")

