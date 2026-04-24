
Language Learning Assistent
(learning by example)

We offer assistance for learning writing in German, Chinese, Japanese, French, Italian, Dutch, Spanish, 
Greek, Portuguese, Russian.

It is assumed that the user is familiar with English.

Any natural language sentence can be represented in different ways,

for example, as a sequence of POS tags:

PRON VERB NOUN ADP DET ADJ CCONJ ADJ NOUN, ADP PRON DET ADP DET NOUN ADJ NOUN CCONJ NOUN ADV VERB AUX AUX. 

If to replace only nouns, verbs and adjectives by POS tags the following reprsentation will be obtained:

jeder VERB NOUN auf eine ADJ und ADJ NOUN in der die in dieser NOUN ADJ NOUN und NOUN VERB werden können. 

Closed-class words are keeped.

Closed class words are a limited set of words that serve specific grammatical functions in sentences. 

Closed class words are essential for constructing grammatically correct sentences.

Such representation better shows sentence structure.

Suppose that the large amount of the grammatically correctly written sentences is available.

Sentences are transformed in some parametric representtion:

PRON VERB NOUN ADP DET ADJ CCONJ ADJ NOUN, ADP PRON DET ADP DET NOUN ADJ NOUN CCONJ NOUN ADV VERB AUX AUX. 

jeder VERB NOUN auf eine ADJ und ADJ NOUN in der die in dieser NOUN ADJ NOUN und NOUN VERB werden können 

Sentences represented in such forms are used to train language model.

The word2vect tool is used to train model.

After training for each sentence is assign a multidimetional vector.

Using trained model such multidimetional vector can be assign to each new sentence (user input sentence).

New sentence can be compared with grammatically correct written sentences from traning set.

This allows user to see correct examples and correct own writing.



How it works

python3 -m venv myenv
source myenv/activate

pip3 install -r requirements.txt


sh model_download.txt

python3  WRITING_CORRECTION_BY_EXAMPLES.py

Steps:

1. Select a language - LANGUAGE

2. Load training data - "Data processing" -> "Load text in selected language from directory ./:
German - de.txt, French - fr.txt and etc"

3. Select number of nearest correct examples -> "Number of nearest examples"

4. Input new sentence to check words order and grammar -> "Write sentence or copy it from clipboard (Ctrl-V)"

5. Tranform sentence in another form -> "Process input sentence"

6. Find other senetences from training set grammatically similar to input sentence -> "Find nearest examples ",

Other options:

If language is selected it is possible to find translation for word written in English or in selected language -> "Search words pair in lexicon"

"Words order errors" - information about words order and typical errors for 10 languages

"Closed-class lexicon" - closed class words words for each language

"2000 words lexicon" - for each language shows 2000 words lexicon with translation


