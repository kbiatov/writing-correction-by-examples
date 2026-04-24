import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import scrolledtext
from tkinter import messagebox
from tkinter import *
import re

import spacy
import os
import subprocess
import sys
import numpy as np
import json

from gensim.models import Word2Vec
from scipy.spatial import distance
import mlpack
from mlpack import  kmeans
from sklearn.cluster import KMeans
import random

from sklearn.neighbors import NearestNeighbors

global language 
global org_language
global poslist
global poswlist
global sentencelist

global input_text
global model
global groups
global nlp

global input_pos_sentence
global input_posword_sentence
global org_sentence
global sentences_vector_list
global output
global lang_en2
global num_clusters
global num_nearest
global centroids
num_clusters = ''
num_nearest = 7

diagnostic_window = None

language = ''
poslist = []
poswlist = []
sentencelist = []
sentences_vector_list = []
#---------------------------------------------------------
def cosine_similarity(vec_a, vec_b):
	    """Calculate cosine similarity between two vectors."""    
	    dot_product = np.dot(vec_a, vec_b)    
	    norm_a = np.linalg.norm(vec_a)    
	    norm_b = np.linalg.norm(vec_b)    
	    return dot_product / (norm_a * norm_b)
#------------------------------------------------
def send_query():
	output_text.delete(1.0, END) # Clear previous output
	output_text.insert(END, "") # Display new result
#-----------------------------------------------
def select_lang(lang):    
	global language  
	global org_language
	global lang_en2
	
	language = lang
	print (language)
	
	if len(language) > 1:
		selected_option = options.get()    
		label.config(text=f"{language}")
		path_name = './WORDS_PAIR.' +language
		#lang_file = language + '_en2'
		lang_en2 = []
		lang_en2 = open_data_file(language,path_name)
		TEXT = ""
#------------------------------------------------------
def clear_window():
	output_text.delete(1.0, END) # Clear previous output
	output_text.insert(END, "") # Display new result
#------------------------------------------------
def save_number(num):
	#global num_clusters
	global num_nearest
	#num_clusters = int(num) 
	num_nearest = int(num)
	#print("Chosen number:", num_clusters)
	print("Num nearest:", num_nearest)
	
#--------------------------------------------
def find_meaning_to():    
	word = entry1.get()   
	word = word.lower() 
	#print (word, lang_en2)
	result = [pair for pair in lang_en2 if word in pair]
	if len(result) > 0:
		
		output = result[0][0] + '    ' + result[0][1]
		print (output)
		entry1.insert(0, "")
		entry1.delete(0, END)   
		entry1.insert(0, output)
	else:
		output = word + '    not found'
		entry1.insert(0, "")
		entry1.delete(0, END)   
		entry1.insert(0, output)
#---------------------------------------
def create_diagnostic_window(dtext):    
	global diagnostic_window    
	if diagnostic_window is None or not diagnostic_window.winfo_exists(): 
	
		diagnostic_window = tk.Toplevel()  # Create a new top-level window        
		diagnostic_window.title("Diagnostic Window")        
		diagnostic_window.geometry("600x50")               
		# Add content to the window        
		label = tk.Label(diagnostic_window, text=dtext,font=('Helvetica', 14))        
		label.pack(pady=10)                
		close_button = tk.Button(diagnostic_window, text="Close", command=diagnostic_window.destroy)            
		close_button.pack(pady=10,font=('Helvetica', 14))
#-----------------------------------------	
def save_history():
	global output
	file_name = language+'_history.txt'
	try:
		fl = open(file_name,'a')
		fl.write(output)
		fl.close()
	except FileNotFoundError:    
		print('File not exist, open new file')  
		fl = open(file_name,'wt+')
		fl.write(output)
		fl.close()
#------------------------------------------
def euclidean_distance(vec1, vec2):
	return np.linalg.norm(np.array(vec1) - np.array(vec2))
#-----------------------------------------------------------
def find_nearest():
	global input_pos_sentence
	global input_posword_sentence
	global model
	global output
	global num_nearest
	
	if language == '':
		create_diagnostic_window('BEFORE PROCESSING NEED TO SELECT LANGUAGE') 
		return 0
		
	if len(sentences_vector_list) == 0:
		create_diagnostic_window('BEFORE PROCESSING NEED TO TRAIN MODEL') 
		return 0
	
	text_area.delete(1.0, END) # Clear previous output
	
	print ('FIND NEAREST',input_pos_sentence)
	print ('FIND NEAREST',input_posword_sentence)
	result = input_pos_sentence
	print (result)
	words = result.split(' ')
	wordlist = []
	for word in words:
		if len(word) > 1:
			wordlist.append(word)
	wordlist = wordlist[:-1]
	sentence_vector = sentence_to_vector(wordlist)
	#print (sentence_vector)
	
	N = num_nearest
	if N > len(sentences_vector_list):
		N = len(sentences_vector_list)
	
	distlist = []
	for vector in sentences_vector_list:
		#dist = euclidean_distance(sentence_vector, vector)
		dist = 1 - cosine_similarity(sentence_vector, vector)
		distlist.append(dist)
	
	indices = np.argpartition(distlist, N)[:N]
	output = ""
	
	ofile = language+'_nearest.txt'
	outfile2 = open(ofile,'wt+')
	outfile2.write((org_sentence))
	outfile2.write('\n\n')
	outfile2.write((input_pos_sentence))
	outfile2.write('\n\n')
	outfile2.write((input_posword_sentence))
	outfile2.write('\n\n')
	outfile2.write('**************************\n')
	
	
	for i in range(N):
		output = output + sentencelist[indices[i]] + '\n\n'
		outfile2.write(sentencelist[indices[i]])
		outfile2.write('\n\n')
		
		output = output + poswlist[indices[i]] + '\n\n'
		outfile2.write(poswlist[indices[i]])
		outfile2.write('\n\n')
		
		output = output + poslist[indices[i]] + '\n'
		outfile2.write(poslist[indices[i]])
		outfile2.write('\n\n')
		
		output = output + '--------------------------\n'
		outfile2.write('--------------------------\n')
		outfile2.write('\n\n')
	
	text_area.delete("1.0","end")
	text_area.insert(END, output,'blue') # Display new result
	outfile2.close()
#------------------------------------------------------------------
def sentence_to_vector(sentence):
	global model
	# Get the word vectors for each word in the sentence
	word_vectors = [model.wv[word] for word in sentence if word in model.wv]
	# Return the average of the word vectors
	return np.mean(word_vectors, axis=0)
#-----------------------------------------------------
def process_sent(language,sentence):
	global nlp
	DASCP = ['DET','ADP','SCONJ', 'CCONJ','PRON','AUX','ACONJ','PREP','PART','PPRON']
	langs_with_gender = ['de','fr','it','es', 'nl', 'el', 'pt', 'ru']
	AVN = ['ADJ', 'VERB', 'NOUN']
	AN = ['ADJ', 'NOUN']
	poses = ''
	posword = ''
	posccw = ''
	doc = nlp(sentence)
	for token in doc:
		#print(language, f'Text: {token.text}, POS: {token.pos_}, Dependency: {token.lemma_}, Head: {token.head.text}')
		if token.pos_ != 'PUNCT':
			poses = poses + token.pos_+ ' '
		else:
			if token.text == ',':
				poses = poses + ', '
			if token.text == '.':
				poses = poses + '. '
			if token.text == '。':
				poses = poses + '. '
				
		if token.pos_ in DASCP:
			#print(language, f'Text: {token.text}, POS: {token.pos_}, Dependency: {token.lemma_}, Head: {token.head.text}')
			posword = posword + token.text.lower() + '   '
			posccw = posccw + token.text.lower() + '   '
			
		else:
			if language == 'ja':
				posword = posword + token.pos_ + '{' + token.lemma_ +'}'+'   '
				posccw = posccw + token.pos_ + '   '
				
			if language == 'zh':
				posword = posword + token.pos_ + '{' + token.text +'}'+'   '
				posccw = posccw + token.pos_ + '   '
				
			else:
				if token.pos_ == 'NOUN':
					last2 = get_end_2chars(token.text)
					gender = gender_detect(language,token.text)
					posword = posword + token.pos_ + '[-' + last2 +'](' + gender + ')   '
					posccw = posccw + token.pos_ + '   '
					
				if token.pos_ == 'ADJ':
					last2 = get_end_2chars(token.text)
					posword = posword + token.pos_ + '[-' + last2 +']   '
					posccw = posccw + token.pos_ + '   '
					
				if token.pos_ == 'ADV':
					last2 = get_end_2chars(token.text)
					posword = posword + token.pos_ + '   '
					posccw = posccw + token.pos_ + '   '
					
				if token.pos_ == 'PROPN':
					last2 = get_end_2chars(token.text)
					posword = posword + token.pos_ + '   '
					posccw = posccw + token.pos_ + '   '
					
				if token.pos_ == 'VERB':
					last3 = get_end_3chars(token.text)
					posword = posword + token.pos_ + '{' + token.lemma_ +'}[-' + last3 +']   '
					posccw = posccw + token.pos_ + '   '
							
	poses = re.sub('SPACE',' ',str(poses))
	poses = re.sub(' +',' ',str(poses))
	poses = re.sub('{}','',str(poses))
	poses = poses + '\n'
	
	posword = re.sub(' {\、}',',',str(posword))
	posword = re.sub('{。}','.',str(posword))
	posword = re.sub('{.}','.',str(posword))
	posword = re.sub('\.','',str(posword))
	
	posword = re.sub('{--}',' ',str(posword))
	posword = re.sub('PUNCT',' ',str(posword))
	posword = re.sub('SPACE',' ',str(posword))
	posword = re.sub(' +',' ',str(posword))
	
	posword = re.sub(' +',' ',str(posword))
	#----------------------------------------
	posccw = re.sub(' {\、}',',',str(posccw))
	posccw = re.sub('{。}','.',str(posccw))
	posccw = re.sub('{.}','.',str(posccw))
	posccw = re.sub('\.','',str(posccw))
	
	posccw = re.sub('{--}',' ',str(posccw))
	posccw = re.sub('PUNCT',' ',str(posccw))
	posccw = re.sub('SPACE',' ',str(posccw))
	posccw = re.sub(' +',' ',str(posccw))
	
	posccw = re.sub(' +',' ',str(posccw))
	#poses = posccw
	return poses,posword,posccw
#----------------------------------------------
def open_file():
	global poslist 
	global poswlist
	global sentencelist
	global nlp
	global language
	
	if language == '':
		text_area.delete(1.0, tk.END)
		create_diagnostic_window('LANGUAGE IS NOT DEFINED, NEED TO SELECT LANGUAGE') 
		return 0
	
	file_path = filedialog.askopenfilename(
		title="Open File",
		filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
	)
	if file_path:
		try:
			with open(file_path, 'r') as file:
				text_area.delete(1.0, tk.END)
				text = file.read()
				#print (text)
				ssent = re.split('\n',text)
				
				load_name = ""
				
				if len(language) > 0 and language != 'zh':
					load_name = language + '_core_news_sm'
					nlp = spacy.load(load_name)
				if len(language) > 0 and language == 'zh':	
					load_name = language + '_core_web_sm'
					nlp = spacy.load(load_name)
				
				if language == '':
					exit(1)
				
				ofile = language+'_train.txt'
				outfile1 = open(ofile,'wt+')
				
				poslist = []
				poswlist = []
				sentencelist = []
				
				ptext = ""
				wtext = ""
				
				for sent in ssent:
					posword = ''
					poses = ''
					oneposes = ''
					poses, posword, oneposes = process_sent(language,sent)
				
					if len(poses) > 25:
						poslist.append(poses)
						poswlist.append(posword)
						sentencelist.append(sent)
						
						outfile1.write((sent))
						outfile1.write('\n\n')
						outfile1.write((poses))
						outfile1.write('\n\n')
						outfile1.write((posword))
						outfile1.write('\n\n')
						outfile1.write((oneposes))
						outfile1.write('\n\n')
						outfile1.write('------------------------------\n')
					
						#ptext = ptext +  poses + '\n\n'
					
						wtext = wtext + sent + '\n\n'
						wtext = wtext + posword + '\n\n'
						wtext = wtext + poses + '\n'
						wtext = wtext + '-------------------------\n'
						
			
				outfile1.close()
				
			text = wtext
			#print (wtext)
			text_area.insert(tk.END, text)
			
			status_label.config(text=f"Opened: {file_path}")
		except:
			status_label.config(text="Error opening file")	
	rez = get_model()
	output_text.delete(1.0, END)
	return 1
#-------------------------------------------------------------
def process_text():
	global nlp
	global input_pos_sentence
	global input_posword_sentence
	global org_sentence
	global poslist
	global poswlist
	
	if language == '':
		create_diagnostic_window('LANGUAGE IS NOT DEFINED, NEED TO SELECT LANGUAGE') 
		return 0
		
		return 0
	if len(poslist) == 0:
		create_diagnostic_window('NEED TO PREPARE TRAINING DATA') 
		return 0
		
	DASCP = ['DET','ADP','SCONJ', 'CCONJ','PRON','AUX','ACONJ','PREP','PART','PPRON']
	langs_with_gender = ['de','fr','it','es', 'nl', 'el', 'pt', 'ru']
	AVN = ['ADJ', 'VERB', 'NOUN']
	AN = ['ADJ', 'NOUN']
	sentence = output_text.get("1.0", "end")    
	doc = nlp(sentence)
	org_sentence = sentence
	posword = ''
	poses = ''
	oneposes = ''

	poses,posword,oneposes = process_sent(language,sentence)
	result = ''
	result = result + sentence + '\n\n'
	result = result + posword + '\n\n'
	result = result + poses + '\n\n'

	input_posword_sentence = posword
	input_pos_sentence = poses
	output_text.delete(1.0, END) # Clear previous output
	output_text.insert(END, result) # Display new result
#-----------------------------------------------
def get_model():
	global model
	global groups
	global poslist
	global language
	global sentences_vector_list
	global lang_en2
	
	if language == '':
		create_diagnostic_window('LANGUAGE IS NOT DEFINED, NEED TO SELECT LANGUAGE') 
		close_button = tk.Button(diagnostic_window, text="Close", command=diagnostic_window.destroy)        
		close_button.pack(pady=10)
		
		return 0
	if len(poslist) == 0:
		create_diagnostic_window('NEED TO PREPARE TRAINING DATA') 
		close_button = tk.Button(diagnostic_window, text="Close", command=diagnostic_window.destroy)        
		close_button.pack(pady=10)
		return 0
	
	sentences = []
	i = 0
	for sentence in poslist:
		#print ('SENT', sentence,'!')
		words = sentence.split(' ')
		words = words[:-1]
		
		wordlist = []
		for word in words:
			wordlist.append(word)
		#wordlist = wordlist[:-1]
		sentences.append(wordlist)
		i = i + 1
	
	model = Word2Vec(sentences, vector_size=25, window=5, min_count=1, workers=4)
	lexicon = model.wv.index_to_key
		
	model_name = language+'_model.dat'
	print (model_name)
	model.save(model_name)
	model = Word2Vec.load(model_name)
	
	sentences_vector_list = []
	for sentence in sentences:
		#print (sentence)
		sentence_vector = sentence_to_vector(sentence)
		sentences_vector_list.append(sentence_vector)
		#print (sentence_vector)
	#return groups
#-------------------------------------------
def open_text_window():   
	TEXT= """ 
	
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

New sentence can be compared with grammatically correct written sentences form traning set.

This allows user to see correct examples and correct own writing.
"""
	# Create top-level window    
	win = tk.Toplevel(root)    
	win.title("Description")    
	win.geometry("1100x1000")       
	# predefined size: width x height    
	win.resizable(False, False)  # fixed size
	# Create scrolled text widget (includes vertical scrollbar)    
	st = scrolledtext.ScrolledText(win, wrap='word',font=('Helvetica', 14))     
	st.pack(fill='both', expand=True, padx=6, pady=6)
	# Insert predefined text and disable editing    
	st.insert('1.0', TEXT)    
	st.configure(state='disabled')
#---------------------------------
def open_text_window1(option):   
	global lang_en2
	if len(option) > 1:
		path_name = './WORDS_PAIR.' +option
		#lang_file = language + '_en2'
		lang_en2 = []
		lang_en2 = open_data_file(option,path_name)
		TEXT = ""
		#print (lang_en2)
		for ln in lang_en2:
			if len(ln) > 1:
				TEXT = TEXT + ln[0].lower() + '    ' + ln[1].lower() + '\n'
	# Create top-level window    
	win = tk.Toplevel(root)    
	info = "Lexicon"
	win.title(info)    
	win.geometry("700x600")       
	# predefined size: width x height    
	win.resizable(False, False)  # fixed size
	# Create scrolled text widget (includes vertical scrollbar)    
	st = scrolledtext.ScrolledText(win, wrap='word',font=('Helvetica', 14))     
	st.pack(fill='both', expand=True, padx=6, pady=6)
	# Insert predefined text and disable editing    
	st.insert('1.0', TEXT)    
	st.configure(state='disabled')
#--------------------------------------------------
def open_text_window2(option):   
	TEXT= """ Frequent Lexicon.""" 
	
	if len(option) > 1:
		path_name = './WORD_ORDER.' +option
		words_order = ''
		info_text = ''
		with open(path_name, 'r') as file:
			info_text = file.read()
		TEXT = ""
		TEXT =  TEXT + info_text + '\n'
	# Create top-level window    
	win = tk.Toplevel(root)    
	info = "Words order"
	win.title(info)    
	win.geometry("900x800")       
	# predefined size: width x height    
	win.resizable(False, False)  # fixed size
	# Create scrolled text widget (includes vertical scrollbar)    
	st = scrolledtext.ScrolledText(win, wrap='word',font=('Helvetica', 14))     
	st.pack(fill='both', expand=True, padx=6, pady=6)
	# Insert predefined text and disable editing    
	st.insert('1.0', TEXT)    
	st.configure(state='disabled')
#-------------------------------------------------
def open_text_window3():   
	TEXT= """ 


How it works

python3 -m venv myenv
source myenv/shopping cart/activate

pip3 install -r requirements.txt

gensim==4.4.0
mlpack==4.7.0
numpy==2.2.6
scikit-learn==1.7.2
scipy==1.15.3
spacy==3.8.14
tk==0.1.0

sh model_download.txt

python -m spacy download de_core_news_sm
python -m spacy download fr_core_news_sm
python -m spacy download it_core_news_sm
python -m spacy download es_core_news_sm
python -m spacy download nl_core_news_sm
python -m spacy download pt_core_news_sm
python -m spacy download ru_core_news_sm
python -m spacy download ja_core_news_sm
python -m spacy download el_core_news_sm
python -m spacy download zh_core_web_sm

python3 LANGUAGE_LEARNING_ASSISTENT.py

Steps:

1. Select a language - LANGUAGE

2. Load training data - "Data processing" -> "Load text in selected language from directory ./:
German - de.txt, French - fr.txt and etc"

3. Select number of nearest correct examples -> "NUmber of nearest examples"

4. Input new sentence to check words order and grammar -> "Write sentence or copy it from clipboard (Ctrl-V)"

5. Tranform sentence in another form -> "Process input sentence"

6. Find other senetences from training set grammatically similar to input sentence -> "Find nearest examples ",

Other options:

If language is selected it is possible to find translation for word written in English or in selected language -> "Search words pair in lexicon"

"Words order errors" - information about words order and typical errors for 10 languages

"Closed-class lexicon" - closed class words words for each language

"2000 words lexicon" - for each language shows 2000 words lexicon with translation

"""
	# Create top-level window    
	win = tk.Toplevel(root)    
	win.title("Description of System")    
	win.geometry("1100x1000")       
	# predefined size: width x height    
	win.resizable(False, False)  # fixed size
	# Create scrolled text widget (includes vertical scrollbar)    
	st = scrolledtext.ScrolledText(win, wrap='word',font=('Helvetica', 13))     
	st.pack(fill='both', expand=True, padx=6, pady=6)
	# Insert predefined text and disable editing    
	st.insert('1.0', TEXT)    
	st.configure(state='disabled')
#------------------------------------------------
def open_text_window4(option):   
	global lang_en2

	if len(option) > 1:
		path_name = './CLCLASS.' +option
		lang_en2 = []
		lang_en2 = open_data_file(option,path_name)
		TEXT = ""
		#print (lang_en2)
		for ln in lang_en2:
			if len(ln) > 1:
				TEXT = TEXT + ln[0].lower() + '    ' + ln[1].lower() + '\n'
				
	# Create top-level window    
	win = tk.Toplevel(root)    
	info = "Lexicon"
	win.title(info)    
	win.geometry("700x600")       
	# predefined size: width x height    
	win.resizable(False, False)  # fixed size
	# Create scrolled text widget (includes vertical scrollbar)    
	st = scrolledtext.ScrolledText(win, wrap='word',font=('Helvetica', 14))     
	st.pack(fill='both', expand=True, padx=6, pady=6)
	# Insert predefined text and disable editing    
	st.insert('1.0', TEXT)    
	st.configure(state='disabled')
#----------------------------------------------------------
def get_end_2chars(s):
		ret = ''
		if len(s) < 2:
			#return "String must be at least 2 characters long."
			return ret
    
		# Split the string into two parts
		main_part = s[:-2]  # All except the last two characters
		last_2 = s[-2:]   # The last two characters
    
		return last_2	
#----------------------------------------------------------
def get_end_3chars(s):
		ret = ''
		if len(s) < 3:
			#return "String must be at least 2 characters long."
			return ret
    
		# Split the string into two parts
		main_part = s[:-3]  # All except the last two characters
		last_3 = s[-3:]   # The last two characters
    
		# Return the parts separated by a hyphen
		return last_3
#----------------------------------------------------
def gender_detect(lang,string):
	outgender = 'x'
	m_de_gender = ['er','en','ig','ling','ismus','or','ant','us']
	f_de_gender = ['in','ion','ung','heit','keit','schaft','ei','ur']
	n_de_gender = ['chen','lein','ment','um','tum']
		
	m_fr_gender = ['age','ment','eau','é','oir','in','on','eur']
	f_fr_gender = ['e','ion','té','esse','ure','ie','ance','ence']
		
	m_it_gender = ['o','ore','ma']
	f_it_gender = ['a','trice','zione']
	v_it_gender = ['e','ista']
		
	m_es_gender = ['o','l','n','r','s','ma']
	f_es_gender = ['a','ción','sión','dad''tad','tud']
	v_es_gender = ['o','a']
		
	m_nl_gender = ['aar','er','eur','or']
	f_nl_gender = ['e','in','es','heid','ing','nis']
	n_nl_gender = ['um','isme','sel','ment']
		
	m_ru_gender = ['б', 'в', 'г', 'д', 'ж', 'з', 'к', 'л', 'м', 'н', 'п', 'р', 'с', 'т', 'ф', 'х', 'ц', 'ч', 'ш', 'щ','й','ь']
	f_ru_gender = ['а','я']
	n_ru_gender = ['о','е','ие']
	v_ru_gender = ['ь']
		
	m_pt_gender = ['o','e']
	f_pt_gender = ['a','ção']
		
	if lang == 'de':
		if any(string.endswith(tuple(m_de_gender)) for ending in string):
			outgender = 'm'
		if any(string.endswith(tuple(f_de_gender)) for ending in string):
			outgender = 'f'
		if any(string.endswith(tuple(n_de_gender)) for ending in string):
			outgender = 'n'
		
	if lang == 'fr':
		if any(string.endswith(tuple(m_fr_gender)) for ending in string):
			outgender = 'm'
		if any(string.endswith(tuple(f_fr_gender)) for ending in string):
			outgender = 'f'
			
	if lang == 'it':
		if any(string.endswith(tuple(m_it_gender)) for ending in string):
			outgender = 'm'
		if any(string.endswith(tuple(f_it_gender)) for ending in string):
			outgender = 'f'
		if any(string.endswith(tuple(v_it_gender)) for ending in string):
			outgender = 'v'
		
	if lang == 'es':
		if any(string.endswith(tuple(m_es_gender)) for ending in string):
			outgender = 'm'
		if any(string.endswith(tuple(f_es_gender)) for ending in string):
			outgender = 'f'
		if any(string.endswith(tuple(v_es_gender)) for ending in string):
			outgender = 'v'
			
	if lang == 'nl':
		if any(string.endswith(tuple(m_nl_gender)) for ending in string):
			outgender = 'm'
		if any(string.endswith(tuple(f_nl_gender)) for ending in string):
			outgender = 'f'
		if any(string.endswith(tuple(n_nl_gender)) for ending in string):
			outgender = 'n'
			
	if lang == 'ru':
		if any(string.endswith(tuple(m_ru_gender)) for ending in string):
			outgender = 'm'
		if any(string.endswith(tuple(f_ru_gender)) for ending in string):
			outgender = 'f'
		if any(string.endswith(tuple(n_ru_gender)) for ending in string):
			outgender = 'n'
		if any(string.endswith(tuple(v_ru_gender)) for ending in string):
			outgender = 'v'
	
	return outgender	
#------------------------------------------------
def open_data_file(language, path_name):
	print (path_name)
	out = []
	with open(path_name) as f:
		for line in f:
			line = re.sub('\n','',str(line))
			split = re.split(' ',line)
			out.append(split)
	return out
#----------------------------------------
def update_selection(event):    
	global language  
	global org_language
	global lang_en2
	
	language = option_var.get()  
	
	language = combo.get() # Capture selection
	if language == 'French':
		language = 'fr'
	if language == 'German':
		language = 'de'
	if language == 'Italian':
		language = 'it'
	if language == 'Spanish':
		language = 'es'
	if language == 'Portuguese':
		language = 'pt'
	if language == 'Greek':
		language = 'el'
	if language == 'Dutch':
		language = 'nl'
	if language == 'Russian':
		language = 'ru'
	if language == 'Japanese':
		language = 'ja'
	if language == 'Chinese':
		language = 'zh'  
	
	if len(language) > 1:
		path_name = './WORDS_PAIR.' +language
		#lang_file = language + '_en2'
		lang_en2 = []
		lang_en2 = open_data_file(language,path_name)
		TEXT = ""
#------------------------------------------------------
# Create window
root = tk.Tk()
root.title("Language Learning Assistent(learining writing by example)")
root.geometry("1200x900")
root.configure(bg="grey")  

# Create menu bar
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

language = ""

entry1 = Entry(root, width=30,font=('Helvetica', 14))
entry1.pack(pady=7, ipady=7) 

button1 = tk.Button(root, text="Search words pair in lexicon",command=find_meaning_to,font=('Helvetica', 14))
button1.place(x=140, y=8)

# Create buttons frame
button_frame = tk.Frame(root)
button_frame.pack(fill=tk.X, padx=5, pady=5)
# Open and process file

options = tk.StringVar()
#options.set("Select an option") 
label = tk.Label(root, text="____",font=('Arial', 14, 'bold'))
label.place(x=20, y=15)

# Help buttons
help_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Menu", menu=help_menu,font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Description", command=lambda: open_text_window(),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="How it works", command=lambda: open_text_window3(),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Exit", command=root.quit,font=('Helvetica', 14,'bold'))

help_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Language", menu=help_menu,font=('Helvetica', 14,'bold'))
help_menu.add_command(label="French", command=lambda: select_lang("fr"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Italian", command=lambda: select_lang("it"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="German", command=lambda: select_lang("de"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Dutch", command=lambda: select_lang("nl"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Spanish", command=lambda: select_lang("es"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Portuguese", command=lambda: select_lang("pt"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Greek", command=lambda: select_lang("el"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Russian", command=lambda: select_lang("ru"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Japanese", command=lambda: select_lang("ja"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Chinese", command=lambda: select_lang("zh"),font=('Helvetica', 14,'bold'))

help_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Data processing", menu=help_menu,font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Load text in selected language", command=open_file,font=('Helvetica', 14,'bold'))

help_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Number of nearest examples", menu=help_menu,font=('Helvetica', 14,'bold'))
help_menu.add_command(label="3", command=lambda:save_number('3'),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="5", command=lambda:save_number(5),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="7", command=lambda:save_number(7),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="10", command=lambda:save_number(10),font=('Helvetica', 14,'bold'))

help_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Words order errors", menu=help_menu,font=('Helvetica', 14,'bold'))
help_menu.add_command(label="French", command=lambda: open_text_window2("fr"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Italian", command=lambda: open_text_window2("it"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="German", command=lambda: open_text_window2("de"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Dutch", command=lambda: open_text_window2("nl"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Spanish", command=lambda: open_text_window2("es"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Portuguese", command=lambda: open_text_window2("pt"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Greek", command=lambda: open_text_window2("el"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Russian", command=lambda: open_text_window2("ru"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Japanese", command=lambda: open_text_window2("ja"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Chinese", command=lambda: open_text_window2("zh"),font=('Helvetica', 14,'bold'))

help_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Closed-class lexicon", menu=help_menu,font=('Helvetica', 14,'bold'))
help_menu.add_command(label="French", command=lambda: open_text_window4("fr"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Italian", command=lambda: open_text_window4("it"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="German", command=lambda: open_text_window4("de"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Dutch", command=lambda: open_text_window4("nl"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Spanish", command=lambda: open_text_window4("es"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Portuguese", command=lambda: open_text_window4("pt"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Greek", command=lambda: open_text_window4("el"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Russian", command=lambda: open_text_window4("ru"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Japanese", command=lambda: open_text_window4("ja"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Chinese", command=lambda: open_text_window4("zh"),font=('Helvetica', 14,'bold'))

help_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="2000 words lexicon", menu=help_menu,font=('Helvetica', 14,'bold'))
help_menu.add_command(label="French", command=lambda: open_text_window1("fr"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Italian", command=lambda: open_text_window1("it"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="German", command=lambda: open_text_window1("de"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Dutch", command=lambda: open_text_window1("nl"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Spanish", command=lambda: open_text_window1("es"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Portuguese", command=lambda: open_text_window1("pt"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Greek", command=lambda: open_text_window1("el"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Russian", command=lambda: open_text_window1("ru"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Japanese", command=lambda: open_text_window1("ja"),font=('Helvetica', 14,'bold'))
help_menu.add_command(label="Chinese", command=lambda: open_text_window1("zh"),font=('Helvetica', 14,'bold'))

text_area = scrolledtext.ScrolledText(root, height=20, width=450,font=('Helvetica', 14))
text_area.pack(pady=10)

tk.Label(root, text="Write sentence or copy it from clipboard (Ctrl-V)",font=('Helvetica', 14,'bold')).place(x=10, y=850)
button3 = tk.Button(root, text="Process input sentence",command=process_text,font=('Helvetica', 14,'bold'))
button3.place(x=450, y=842)
button4 = tk.Button(root, text="Find nearest examples ",command=find_nearest,font=('Helvetica', 14,'bold'))
button4.place(x=700, y=842)
button5 = tk.Button(root, text="Clear window ",command=clear_window,font=('Helvetica', 14,'bold'))
button5.place(x=960, y=842)

# Output text area
output_text = scrolledtext.ScrolledText(root, height=15, width=400,font=('Helvetica', 14))
output_text.place(x=100, y=480) 
output_text.pack(pady=10)

# Status label
status_label = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_label.pack(side=tk.BOTTOM, fill=tk.X)

root.mainloop()
