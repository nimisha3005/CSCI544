import numpy as np
import json
import sys

class HMM:
    def __init__(self) -> None:
        self.tags = set()
        self.words = set()
        self.tag_count = {}

        self.emmission = dict()
        self.transition = dict()



    def read_data(self, filename):
        with open(filename, 'r') as f:
            lines = f.read().splitlines()
        return lines


    def get_tokens(self, data):
        tokens = []
        for line in data:
            tokens.extend(line.split())
        return tokens

    
    def get_words(self, tokens):
        for word in tokens:
            self.words.add("/".join(word.split("/")[:-1]))          

    
    def get_tags(self, tokens):
        self.tags.add("<start>")
        for word in tokens:
            self.tags.add(word.split("/")[-1])
            #add tag to tag_count dictionary
            if word.split("/")[-1] not in self.tag_count:
                self.tag_count[word.split("/")[-1]] = 1
            else:    
                self.tag_count[word.split("/")[-1]] += 1


    def create_emission(self, tokens):
        
        # for each word in words list create dictionary with tag from tags list as key and their count of occurence as value   
        emission = {word: {tag: 1e-10 for tag in self.tags} for word in self.words}

        for token in tokens:       
            emission["/".join(token.split("/")[:-1])][token.split("/")[-1]] += 1
 
        # each tag in emission dictionary is divided by the number of occurences of that tag in the corpus
        for word in self.words:
            for tag in self.tags:
                if tag!="<start>":
                    emission[word][tag] /= (self.tag_count[tag] + 1e-10)
       
        self.emmission = emission

    
    def create_transition(self, lines):

        transition = {tag: {tag: 1e-10 for tag in self.tags} for tag in self.tags}

        for line in lines:
            i=0
            l = line.split()
            for j in range(len(l)):
                if i==0:
                    transition["<start>"][l[j].split("/")[-1]] += 1
                else:
                    transition[l[j-1].split("/")[-1]][l[j].split("/")[-1]] += 1              
                i+=1
                
        for tag in transition:
            total = sum(transition[tag].values())
            for tag2 in transition[tag]:
                transition[tag][tag2] /= (total + 1e-10)
        
        self.transition = transition

    

if __name__ == "__main__":
    input = sys.argv[1]
    model = HMM()
    lines = model.read_data(input)

    tokens = model.get_tokens(lines)

    model.get_tags(tokens)
    model.get_words(tokens)
    model.create_emission(tokens)

    model.create_transition(lines)

    with open('hmmmodel.txt', 'w', encoding='utf-8') as txt_file:
        txt_file.write(json.dumps({"emission": model.emmission, "transition": model.transition, "tags": list(model.tags)}, ensure_ascii=False))
