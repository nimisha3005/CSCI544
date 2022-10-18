import sys
import re
import json

class Preprocess_Data:
    def __init__(self):
        self.ids = list()
        self.true_fake = list()
        self.pos_neg = list()
        self.reviews = list()
        self.stopwords = list()

        self.words = dict()
        self.weights1 = dict()
        self.weights2 = dict()
        self.average_weights1 = dict()
        self.average_weights2 = dict()
        self.bias1 = 0
        self.bias2 = 0

    
    def read_data(self,filename):
        """
        Read the data from the file and store it in the class variables
        """
        with open(filename, 'r') as f:
            lines = f.read().splitlines()

        for row in lines:
            row_data = row.split(" ")
            self.ids.append(row_data[0])
            self.true_fake.append(1 if row_data[1]=="True" else -1)
            self.pos_neg.append(1 if row_data[2]=="Pos" else -1)
            self.reviews.append(" ".join(row_data[i].lower() for i in range(3,len(row_data))))
   
        self.remove_punctuation()
        self.remove_stopwords()

        self.word_count()
    
    
    def remove_stopwords(self):
        """
        Remove the stopwords from the reviews
        """
        self.stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "so", "than", "too", "very", "can", "will", "just", "should", "now"]
        self.stop_words = set(self.stopwords)
        for i in range(len(self.reviews)):
            self.reviews[i] = " ".join([word for word in self.reviews[i].split() if word not in self.stop_words])
    

    def remove_punctuation(self):
        """
        Remove the punctuation from the reviews using regex
        """
        # remove punctuation using regex
        for i in range(len(self.reviews)):
            self.reviews[i] = re.sub(r'[^\w\s]','',self.reviews[i])


    def word_count(self):
        """
        Count the number of times a word appears in the reviews and initialize the weights
        """
        for review in self.reviews:
            for word in review.split():
                if word not in self.words:
                    self.words[word] = 1
                    self.weights1[word] = 0
                    self.weights2[word] = 0
                    self.average_weights1[word] = 0
                    self.average_weights2[word] = 0
                else:
                    self.words[word] += 1


### Vanilla Perceptron Algorithm
class VanillaPerceptron:
    def __init__(self,data):
        self.epochs = 10

        self.vanilla_weights1 = data.weights1
        self.vanilla_weights2 = data.weights2
        self.vanilla_bias1 = data.bias1
        self.vanilla_bias2 = data.bias2


    def train(self, data):
        """
        Train the model on the training data and update the weights
        """
        for _ in range(self.epochs):
            for row in range(len(data.reviews)):
                review = data.reviews[row]
                true_fake = data.true_fake[row]
                pos_neg = data.pos_neg[row]

                words = dict()
                for word in review.split():
                    if word=="": continue
                    if word not in words:
                        words[word] = 1
                    else:
                        words[word] += 1

                self.update_weights(words, true_fake, pos_neg)


    def update_weights(self, review, true_fake, pos_neg):
        """
        Update activation values and bias for each word in the review
        """
        self.vanilla_activation1 = 0
        self.vanilla_activation2 = 0

        for word in review:
            self.vanilla_activation1 += self.vanilla_weights1[word] * review[word]
        self.vanilla_activation1 += self.vanilla_bias1

        if true_fake * self.vanilla_activation1 <= 0:
            for word in review:
                self.vanilla_weights1[word] += true_fake * review[word]
            
            self.vanilla_bias1 += true_fake


        for word in review:
            self.vanilla_activation2 += self.vanilla_weights2[word] * review[word]
        self.vanilla_activation2 += self.vanilla_bias2
        
        if pos_neg * self.vanilla_activation2 <= 0:
            for word in review:
                self.vanilla_weights2[word] += pos_neg * review[word]
            
            self.vanilla_bias2 += pos_neg     


    def predict(self, review):
        """
        Predict the class of the review
        """
        prediction = 0
        for word in review.split():
            if word in self.vanilla_weights1:
                prediction += self.vanilla_weights1[word]
        prediction += self.vanilla_bias1
        return prediction


    def test(self, data):
        """
        Test the model on the test data
        """
        correct = 0
        for i in range(len(data.reviews)):
            review = data.reviews[i]
            true_fake = data.true_fake[i]
            
            prediction = self.predict(review)
            if prediction > 0 and true_fake == 1:
                correct += 1
            elif prediction < 0 and true_fake == -1:
                correct += 1
        return correct / len(data.reviews)


### Averaged Perceptron Algorithm
class AveragePerceptron:
    def __init__(self,data):
        self.weights = dict()
        self.epochs = 80

        self.average_weights1 = data.average_weights1
        self.average_weights2 = data.average_weights2
        self.average_bias1 = data.bias1
        self.average_bias2 = data.bias2
        self.weights1 = data.weights1
        self.weights2 = data.weights2

        self.count = 1
        self.beta1 = 0
        self.beta2 = 0
    

    def train(self, data):
        """
        Train the model on the training data and update the weights
        """
        for _ in range(self.epochs):
            for row in range(len(data.reviews)):
                review = data.reviews[row]
                true_fake = data.true_fake[row]
                pos_neg = data.pos_neg[row]

                words = dict()
                for word in review.split():
                    if word=="": continue
                    if word not in words:
                        words[word] = 1
                    else:
                        words[word] += 1

                self.update_weights(words, true_fake, pos_neg)
        
        self.update_average_weights_bias()


    def update_weights(self, review, true_fake, pos_neg):
        """
        Update activation values and bias for each word in the review
        Update average weights and beta values
        """
        self.activation1 = 0
        self.activation2 = 0

        for word in review:
            self.activation1 += self.weights1[word] * review[word]
        self.activation1 += self.average_bias1

        if true_fake * self.activation1 <= 0:
            for word in review:
                self.weights1[word] += true_fake * review[word]
                self.average_weights1[word] += self.count * true_fake * review[word]
            
            self.average_bias1 += true_fake
            self.beta1 += self.count * true_fake



        for word in review:
            self.activation2 += self.weights2[word] * review[word]
        self.activation2 += self.average_bias2
        
        if pos_neg * self.activation2 <= 0:
            for word in review:
                self.weights2[word] += pos_neg * review[word]
                self.average_weights2[word] += self.count * pos_neg * review[word]
            
            self.average_bias2 += pos_neg
            self.beta2 += self.count * pos_neg
        
        self.count += 1


    def update_average_weights_bias(self):
        """
        Update average weights and bias for each word
        """
        for word in self.weights1:
            self.average_weights1[word] = self.weights1[word] - (self.average_weights1[word] / float(self.count))
        
        self.average_bias1 = self.average_bias1 - (self.beta1 / self.count)


        for word in self.weights2:
            self.average_weights2[word] = self.weights2[word] - (self.average_weights2[word] / float(self.count))

        self.average_bias2 = self.average_bias2 - (self.beta2 / self.count)


    def predict(self, review):
        """
        Predict the class of the review
        """
        prediction = 0
        for word in review.split():
            if word in self.average_weights1:
                prediction += self.average_weights1[word]
        prediction += self.average_bias1
        return prediction


    def test(self, data):
        """
        Test the model on the test data
        """
        correct = 0
        for i in range(len(data.reviews)):
            review = data.reviews[i]
            true_fake = data.true_fake[i]
            
            prediction = self.predict(review)
            if prediction > 0 and true_fake == 1:
                correct += 1
            elif prediction < 0 and true_fake == -1:
                correct += 1
        return correct / len(data.reviews)
            


if __name__ == "__main__":
    input_file = sys.argv[1]

    preprocessObj = Preprocess_Data()
    preprocessObj.read_data(input_file)

    # Train the vanilla perceptron model
    vanillaPerceptronObj = VanillaPerceptron(preprocessObj)
    vanillaPerceptronObj.train(preprocessObj)

    with open('vanillamodel.txt', 'w', encoding='utf-8') as txt_file:
            txt_file.write(json.dumps({"words":preprocessObj.words, "weights1":vanillaPerceptronObj.vanilla_weights1, "weights2":vanillaPerceptronObj.vanilla_weights2, "bias1":vanillaPerceptronObj.vanilla_bias1, "bias2":vanillaPerceptronObj.vanilla_bias2, "stopwords":list(preprocessObj.stop_words)}, ensure_ascii=False))


    # Train the averaged perceptron model
    averagePerceptronObj = AveragePerceptron(preprocessObj)
    averagePerceptronObj.train(preprocessObj)

    with open('averagedmodel.txt', 'w', encoding='utf-8') as txt_file:
        txt_file.write(json.dumps({"words":preprocessObj.words, "weights1":averagePerceptronObj.average_weights1, "weights2":averagePerceptronObj.average_weights2, "bias1":averagePerceptronObj.average_bias1, "bias2":averagePerceptronObj.average_bias2, "stopwords":list(preprocessObj.stop_words)}, ensure_ascii=False))