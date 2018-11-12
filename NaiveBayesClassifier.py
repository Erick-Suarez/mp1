# -*- coding: utf-8 -*-
import os, sys, time, math, string

# ======================================================================
# Main Method
# ======================================================================

def main():
    trainingFileName = "training.txt"
    testingFileName = "testing.txt"

    documentClassifier = NaiveBayesClassifier()

    trainStart = time.time()
    documentClassifier.trainWithDataSet(trainingFileName)
    trainEnd = time.time()
    trainTime = str(trainEnd - trainStart) + "seconds (training)"

    labelStart = time.time()
    documentClassifier.createModel()
    labelEnd = time.time()
    labelTime = str(labelEnd - labelStart) + "seconds (labeling)"

    trainingAccuracy = documentClassifier.classifyDataSet(trainingFileName, "training")
    testingAccuracy = documentClassifier.classifyDataSet(testingFileName, "testing")

    print(trainTime)
    print(labelTime)
    print(trainingAccuracy)
    print(testingAccuracy)

# ======================================================================
# Naive Bayes Classifier Class
# ======================================================================

class NaiveBayesClassifier:
    def __init__(self):
        self.numberOfAuthors = 15
        self.classifiedDocuments = {}
        self.authors = []
        self.authorProbabilities = []
        self.authorTotalWords = []
        self.documentLengths = {}
        self.condProbabilities = {}
        self.numOfDocumentsWordAppearsIn = {}

        for i in range(16):
            self.authorProbabilities.append(0.0)

        for i in range(16):
            self.authorTotalWords.append(0)

        for i in range(16):
            self.authors.append([])

    def trainWithDataSet(self, fileName):
        file = open(fileName, "r", encoding = "ISO-8859-1")
        lines = file.read().split('\n')

        self.initializeCondProbabilityMap(lines)
        
        # Populate Document Word Frequency Maps
        documentNumber = 0
        for line in lines:
            line = line.split(",")

            if len(line) == 2:
                content = line[0]
                author = line[1]

                self.authors[int(author)].append(documentNumber)

                self.classifiedDocuments[documentNumber] = {}

                words = self.tokenize(content)
                self.populateWordFrequencyMap(words, self.classifiedDocuments[documentNumber])
                
                for word in self.classifiedDocuments[documentNumber]:
                    if word in self.numOfDocumentsWordAppearsIn:
                        self.numOfDocumentsWordAppearsIn[word] += 1
                    else:
                        self.numOfDocumentsWordAppearsIn[word] = 1

                self.documentLengths[documentNumber] = len(words)
                documentNumber += 1

        file.close()

    def createModel(self):
        # Calculate and store author probabilities
        authorIndex = 1
        while(authorIndex < len(self.authorProbabilities)):
            self.authorProbabilities[authorIndex] = self.probability(authorIndex)
            authorIndex += 1
        
        # Store total words for author
        # for authorIndex in range(1, len(self.authorProbabilities)):
        #     totalWords = 0
        #     for word in self.condProbabilities:
        #         for documentNumber in self.authors[int(authorIndex)]:
        #             if word in self.classifiedDocuments[documentNumber]:
        #                totalWords += self.classifiedDocuments[documentNumber][word]
        #     self.authorTotalWords[authorIndex] = totalWords

        # Calculate and store conditional probabilities
        for word in self.condProbabilities:
            authorIndex = 1
            while(authorIndex < len(self.authorProbabilities)):
                condProb = self.condProbability(word, authorIndex)
                if condProb != 0:
                    self.condProbabilities[word][authorIndex] = condProb
                authorIndex += 1
        
        print("Modeling Done.")
        
    def classifyDataSet(self, fileName, dataSetType):
        print("Classifying Data Set...")

        file = open(fileName, "r", encoding = "ISO-8859-1")
        accuracy = 0.0
        documentNumber = 0

        lines = file.read().split('\n')

        for line in lines:
            line = line.split(" ,")
            
            if len(line) == 2:
                content = line[0]
                author = line[1]

                words = self.tokenize(content)
                likelyClassification = self.classifyDocument(words)
                if(dataSetType == "testing"):
                    print(likelyClassification)

                if(likelyClassification == author):
                    accuracy += 1.0
                documentNumber += 1

        file.close()
        
        return str(accuracy / len(self.classifiedDocuments)) + " (" + dataSetType + ")"

# ======================================================================
# Helper Functions
# ======================================================================
  
    def classifyDocument(self, words):
        maxProbability = -1000000000
        likelyClassification = 0

        uniqueWords = {}
        tfIDFUniqueWords = {}
        self.populateWordFrequencyMap(words, uniqueWords)
        self.populateTFIDFMap(len(words), tfIDFUniqueWords, uniqueWords)
        authorIndex = 1

        while(authorIndex < len(self.authors)):
            if len(self.authors[authorIndex]) > 0:
                currProbability = 1                
                for word in tfIDFUniqueWords:
                    if tfIDFUniqueWords[word] > 0 and word in self.condProbabilities:
                          currProbability += (self.condProbabilities[word][authorIndex])
                currProbability += self.authorProbabilities[authorIndex]
                if currProbability > maxProbability:
                    maxProbability = currProbability
                    likelyClassification = str(authorIndex)

            authorIndex += 1
        return likelyClassification

    def condProbability(self, word, author):
        probability = 1

        for documentNumber in self.authors[int(author)]:
            print("==================")
            print(self.classifiedDocuments[documentNumber])

            if word in self.classifiedDocuments[documentNumber]:
                probability += self.classifiedDocuments[documentNumber][word] 
        
        probability += 1

        totalWords = self.authorTotalWords[author]

        probability = math.log(probability, 2) - math.log((totalWords + len(self.condProbabilities)), 2)
        
        # return probability

        # for documentNumber in self.authors[int(author)]:
        #     if word in self.classifiedDocuments[documentNumber]:
        #         probability += float(self.classifiedDocuments[documentNumber][word])/self.documentLengths[documentNumber]
        #     else:
        #         probability += 1.0/len(self.classifiedDocuments[documentNumber])

        #     probability /= float(len(self.authors[int(author)]))
   
        # return math.log(probability, 2)

    def probability(self, author):
        return math.log((len(self.authors[author])*1.0)/len(self.classifiedDocuments), 2)

    def populateWordFrequencyMap(self, words, wordFrequencyMap):
        if words is not None:
            for word in words:
                if word in wordFrequencyMap: 
                    wordFrequencyMap[word] = wordFrequencyMap[word] + 1
                else:
                    wordFrequencyMap[word] = 1


    def populateTFIDFMap(self, wordsLength, tfIDFMap, wordFrequencyMap):

        for word in wordFrequencyMap:
            tf = wordFrequencyMap[word]/wordsLength
            
            idf = len(self.classifiedDocuments)

            count = self.numOfDocumentsWordAppearsIn[word]

            if(count == 0):
                count = 1
            idf /= count

            tfIDFMap[word] = tf*(1 + math.log(idf, 10))

    def initializeCondProboabilityMap(self, lines):
        for line in lines:
            line = line.split(",")

            if len(line) == 2:
                content = line[0]

                words = self.tokenize(content)

                for word in words:
                    if word not in self.condProbabilities:
                        authorsList = []
                        for i in range(16):
                            authorsList.append(0.0)
                            self.condProbabilities[word] = authorsList

    def tokenize(self, line):
        if line is not None:
            # Split by whitespace and normalize
            words = line.lower().split()
            return words
        else:
            return None


main()
