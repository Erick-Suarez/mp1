# -*- coding: utf-8 -*-
import os, sys, time, math, string, operator

# ======================================================================
# Main Method
# ======================================================================

def main():
    fileArgs = sys.argv
    trainingFileName = fileArgs[1]
    testingFileName = fileArgs[2]

    documentClassifier = NaiveBayesClassifier()

    trainStart = time.time()
    documentClassifier.trainWithDataSet(trainingFileName)
    trainEnd = time.time()
    trainTime = str(trainEnd - trainStart) + " seconds (training)"

    labelStart = time.time()
    documentClassifier.createModel()
    labelEnd = time.time()
    labelTime = str(labelEnd - labelStart) + " seconds (labeling)"

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

        self.classifiedDocumentsBigram = {}
        self.condProbabilitiesBigram = {}
        self.numOfDocumentsBigramAppearsIn = {}

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
        # self.initializeBigramCondProbabilityMap(lines)

        # Populate Document Word Frequency Maps
        documentNumber = 0
        for line in lines:
            line = line.split(",")

            if len(line) == 2:
                content = line[0]
                author = line[1]

                self.authors[int(author)].append(documentNumber)

                self.classifiedDocuments[documentNumber] = {}
                self.classifiedDocumentsBigram[documentNumber] = {}

                words = self.tokenize(content)
                wordsBigram = self.tokenizeBigram(content)

                self.populateWordFrequencyMap(words, self.classifiedDocuments[documentNumber])
                # self.populateWordFrequencyMap(wordsBigram, self.classifiedDocumentsBigram[documentNumber])

                for word in self.classifiedDocuments[documentNumber]:
                    if word in self.numOfDocumentsWordAppearsIn:
                        self.numOfDocumentsWordAppearsIn[word] += 1
                    else:
                        self.numOfDocumentsWordAppearsIn[word] = 1
                # for bigram in self.classifiedDocumentsBigram[documentNumber]:
                #     if bigram in self.numOfDocumentsBigramAppearsIn:
                #         self.numOfDocumentsBigramAppearsIn[bigram] += 1
                #     else:
                #         self.numOfDocumentsBigramAppearsIn[bigram] = 1

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
        for authorIndex in range(1, len(self.authorProbabilities)):
            totalWords = 0
            for word in self.condProbabilities:
                for documentNumber in self.authors[int(authorIndex)]:
                    if word in self.classifiedDocuments[documentNumber]:
                       totalWords += self.classifiedDocuments[documentNumber][word]
            self.authorTotalWords[authorIndex] = totalWords

        # Calculate and store conditional probabilities
        for word in self.condProbabilities:
            authorIndex = 1
            while(authorIndex < len(self.authorProbabilities)):
                condProb = self.condProbability(word, authorIndex)
                if condProb != 0:
                    self.condProbabilities[word][authorIndex] = condProb
                authorIndex += 1

        # Calculate and store bigram conditional probabilities
        # for bigram in self.condProbabilitiesBigram:
        #     authorIndex = 1
        #     while(authorIndex < len(self.authorProbabilities)):
        #         condProb = self.condProbabilityBigram(bigram, authorIndex)
        #         if condProb != 0:
        #             self.condProbabilitiesBigram[bigram][authorIndex] = condProb
        #         authorIndex += 1
        #print(self.condProbabilitiesBigram)

    def classifyDataSet(self, fileName, dataSetType):

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
                # bigrams = self.tokenizeBigram(content)

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
        # uniqueBigrams = {}
        tfIDFUniqueWords = {}
        self.populateWordFrequencyMap(words, uniqueWords)
        # self.populateWordFrequencyMap(bigrams, uniqueBigrams)
        self.populateTFIDFMap(len(words), tfIDFUniqueWords, uniqueWords)
        authorIndex = 1

        while(authorIndex < len(self.authors)):
            if len(self.authors[authorIndex]) > 0:
                currProbability = 1

                topTwentyPercent = len(tfIDFUniqueWords)/3
                topTwentyPercentWords = sorted(tfIDFUniqueWords.items(), key=operator.itemgetter(1), reverse=True)
                count = 0
                for word in topTwentyPercentWords:
                    if(count > topTwentyPercent):
                        break
                    if word[0] in self.condProbabilities:
                        currProbability += self.condProbabilities[word[0]][authorIndex]
                    else:
                        currProbability += math.log(1.0/len(self.condProbabilities), 2)
                    count+=1
                # for bigram in uniqueBigrams:
                #     if bigram in self.condProbabilitiesBigram:
                #         currProbability += self.condProbabilitiesBigram[bigram][authorIndex]
                currProbability += self.authorProbabilities[authorIndex]
                if currProbability > maxProbability:
                    maxProbability = currProbability
                    likelyClassification = str(authorIndex)

            authorIndex += 1
        return likelyClassification

    def condProbability(self, word, author):
        probability = 1

        for documentNumber in self.authors[int(author)]:
            if word in self.classifiedDocuments[documentNumber]:
                probability += self.classifiedDocuments[documentNumber][word]

        totalWords = self.authorTotalWords[author]

        probability = math.log(probability, 2) - math.log((totalWords + len(self.condProbabilities)), 2)

        return probability

        # probability = 0
        # for documentNumber in self.authors[int(author)]:
        #     if word in self.classifiedDocuments[documentNumber]:
        #         probability += float(self.classifiedDocuments[documentNumber][word])/self.documentLengths[documentNumber]
        #     else:
        #         probability += 1.0/len(self.classifiedDocuments[documentNumber])
        #
        #     probability /= float(len(self.authors[int(author)]))
        #
        # return math.log(probability, 2)

    def condProbabilityBigram(self, bigram, author):
        probability = 1

        for documentNumber in self.authors[int(author)]:
            if bigram in self.classifiedDocumentsBigram[documentNumber]:
                probability += self.classifiedDocumentsBigram[documentNumber][bigram]

        totalWords = self.authorTotalWords[author]/2.0

        probability = math.log(probability, 2) - math.log((totalWords + len(self.condProbabilitiesBigram)), 2)

        return probability

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

            count = 1 + self.numOfDocumentsWordAppearsIn[word]

            idf /= count

            tfIDFMap[word] = tf*math.log(idf, 10)

    def initializeCondProbabilityMap(self, lines):
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
    def initializeBigramCondProbabilityMap(self, lines):
        for line in lines:
            line = line.split(",")

            if len(line) == 2:
                content = line[0]

                bigrams = self.tokenizeBigram(content)

                for bigram in bigrams:
                    if bigram not in self.condProbabilitiesBigram:
                        authorsList = []
                        for i in range(16):
                            authorsList.append(0.0)
                        self.condProbabilitiesBigram[bigram] = authorsList

    def tokenize(self, line):
        if line is not None:
            # Split by whitespace and normalize
            words = line.lower().split()
            return words
        else:
            return None

    def tokenizeBigram(self, line):
        if line is not None:
            # Split by every two whitespaces
            span = 2
            ogWords = line.lower().split()

            words = ogWords
            words = [" ".join(words[i:i+span]) for i in range(0, len(words), span)]

            words += [" ".join(ogWords[i:i+span]) for i in range(1, len(ogWords))]
            return words
        else:
            return None


main()
