import numpy as np
import json
import sys

class Viterbi:
    def __init__(self) -> None:
        self.data = None
        self.emission = None
        self.transition = None
        self.tags = None

        self.result_matrix = ""


    def read_data(self, filename):
        with open(filename, 'r') as f:
            json_data = json.load(f)
        self.data = json_data
        self.emission = json_data["emission"]
        self.transition = json_data["transition"]
        self.tags = json_data["tags"]


    def viterbi_algorithm(self,sentence):

        viterbi_matrix = {}
        backpointer = {}

        for tag in self.tags:
            viterbi_matrix[tag] = {}
            backpointer[tag] = {}

            if sentence[0] in self.emission:
                viterbi_matrix[tag][0] = self.transition["<start>"][tag] * self.emission[sentence[0]][tag]
            else:
                viterbi_matrix[tag][0] = self.transition["<start>"][tag]
            
            backpointer[tag][0] = "<start>"


        for word_index in range(1, len(sentence)):
            for tag in self.tags:
    
                if sentence[word_index] in self.emission:  
                    prob = [viterbi_matrix[prev_tag][word_index - 1] * self.transition[prev_tag][tag] * self.emission[sentence[word_index]][tag] for prev_tag in self.tags]

                    viterbi_matrix[tag][word_index] = max(prob)
                    backpointer[tag][word_index] = self.tags[np.argmax(prob)]
                else:
                    prob = [viterbi_matrix[prev_tag][word_index - 1] * self.transition[prev_tag][tag] for prev_tag in self.tags]

                    viterbi_matrix[tag][word_index] = max(prob)
                    backpointer[tag][word_index] = self.tags[np.argmax(prob)]


        prob_max = 0.0
        tag_max = None

        for tag in self.tags:
            if viterbi_matrix[tag][len(sentence) - 1] > prob_max:
                prob_max = viterbi_matrix[tag][len(sentence) - 1]
                tag_max = tag
            
        
        #backpropagate to find the most likely sequence of tags using backpointer
        result = []
        result.append(tag_max)
        for word_index in range(len(sentence) - 1, 0, -1):
            result.append(backpointer[result[-1]][word_index])
            
        result.reverse()

        for i in range(len(sentence)):
            if i!=len(sentence)-1:
                self.result_matrix += sentence[i] + "/" + result[i] + " "
            else:
                self.result_matrix += sentence[i] + "/" + result[i] + "\n"

    
    def write_result(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.result_matrix)
        

if __name__ == "__main__":
    input_file = sys.argv[1]
    viterbi_model = Viterbi()

    viterbi_model.read_data("hmmmodel.txt")


    with open(input_file, 'r') as f:
        lines = f.read().splitlines()

        for line in lines:        
            viterbi_model.viterbi_algorithm(line.split())
    
    viterbi_model.write_result("hmmoutput.txt")
