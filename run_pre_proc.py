from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import json

from utils import printLines, loadLinesAndConversations, extractSentencePairs
from utils import Voc, loadPrepareData, trimRareWords, batch2TrainData, MIN_COUNT

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

corpus_name = "movie-corpus"
corpus = os.path.join("data", corpus_name)

# printLines(os.path.join(corpus, "utterances.jsonl"),1)

# Define path to new file
datafile = os.path.join(corpus, "formatted_movie_lines.txt")

delimiter = '\t'
# Unescape the delimiter
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

# Initialize lines dict and conversations dict
lines = {}
conversations = {}
# Load lines and conversations
print("\nProcessing corpus into lines and conversations...")
lines, conversations = loadLinesAndConversations(os.path.join(corpus, "utterances.jsonl"))

# Write new csv file
print("\nWriting newly formatted file...\n")
with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
    for pair in extractSentencePairs(conversations):
        writer.writerow(pair)

# Print a sample of lines
"""
print("\nSample lines from file: \n")
printLines(datafile)
"""


# Load/Assemble voc and pairs
"""
save_dir = os.path.join("data", "save")
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)

# Print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)

# Trim voc and pairs
pairs = trimRareWords(voc, pairs, MIN_COUNT)
"""

# Example for validation by picking
"""
small_batch_size = 5 # for demonstration pair batch size is taken to be 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)
"""


