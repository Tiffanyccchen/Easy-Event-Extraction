#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Preinstall packages
--------------
l. opencc : a tool for converting traditional Chinese charaters into simplified Chinese charaters.
https://github.com/yichen0831/opencc-python

2. jieba : an efficient Chinese tokenization tool.
https://github.com/fxsjy/jieba

3. pyltp : a traditional Chinese language preprocessing utility containing tokenization, p.o.s. tagging, ne recognition , dependency parsing, etc. tools.
https://github.com/HIT-SCIR/pyltp

After Installation, you should have a ltp_data folder containing parser.model and pos.model ready for loading
'''
 
'''Usage examples
--------------
Extract events from Chinese sentences
    >>> from Event_Extraction import Event_Extraction
    >>> events = Event_Extraction('三商美邦总经理杨棋材，请辞获准。', tra_sim = True, tokenize = False, expand = True)
    >>> events = Event_Extraction('三商 美邦 总经理 杨棋材 ，请辞 获准 。', tra_sim = True, tokenize = True, expand = True)
    >>> events = Event_Extraction('三商 美邦 總經理 楊棋材 ，请辭 獲准 。', tra_sim = False, tokenize = True, expand = False)
    
    Parameters
    -----
    tra_sim : 輸入字句是中文還是英文
    tokenize : 有無預先分詞好
    output_tra : 輸出要繁體或簡體 

You can then infer events on a new, unseen corpus. There are two types of events extraction : standard version and expand version
    >>> events.events
    standard version:
    >>> [['总经理','杨棋材'], ['请辞','获准']]
    expand version:
    [['美邦','总经理','杨棋材'], ['请辞','获准']]
'''

import time
import re
import os
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from opencc import OpenCC
ts = OpenCC('t2s')
st = OpenCC('s2t')

import jieba

from pyltp import Postagger
postagger = Postagger()
postagger.load("./pre_tools/ltp_data/pos.model")

from pyltp import Parser
parser = Parser()
parser.load("./pre_tools/ltp_data/parser.model")


class Event_extraction():

    def __init__(self, corpus = None, expand = False, tra = True, tokenize = True, output_tra = True):
        """
        Parameters
        ----------
        corpus : string, (tokenized) (simplified) sentences with punctuations
        tra : boolean, whether sentences are composed of simplified/traditional characters
        tokenize : boolean, whether sentences are tokenized (separated by space) or not
        output_tra: boolean, whether you want the events composed of traditional/ simplified characters
        """
        # check if corpus is passed
        if corpus is None :
            raise ValueError(
                'at least one sentence must be specified, to extract events'
            )
        self.corpus = corpus

        # translate to simplified, tokenize, sentence split, remove null
        self.corpus = list(re.split('，|。|！|；|：|？', self.corpus)) 
         
        if not tra:
            self.corpus = [self._tra_to_sim(sent)  for sent in self.corpus]

        if not tokenize:
            self.corpus = [' '.join(list(jieba.cut(sent))) for sent in self.corpus]

        self.corpus = [sent.split(' ') for sent in self.corpus]
        self.corpus =[list(filter(lambda x: x != '',sent)) for sent in self.corpus]
        self.corpus = [sent for sent in self.corpus if sent != []]

        # obtain pos, dep, and events
        self.pos = [list(postagger.postag(sent)) for sent in self.corpus]
        self.dep = self.dependency_parsing(self.corpus, self.pos)
        self.events = self.event_extraction(self.corpus, self.pos, self.dep, expand)
        self.events_idxes =  self.events[1]
        if not output_tra:
            self.events = self.events[0]
        else:
            self.events = [st.convert(' '.join(event)).split(' ') for event in self.events[0]]

    def _tra_to_sim(self,corpus):
        '''
        translate traditional characters to simplified ones
        Parameters
        ----------
        corpus: string, traditional Chinese sentences separated by Chinese punctuations

        Returns
        -------
        corpus: string, simplified Chinese sentences separated by Chinese punctuations
        '''
        return ts.convert(str(corpus))

    def _sim_to_tra(self,event):
        '''
        translate simplified characters to traditional ones
        Parameters
        ----------
        event: string, simplified Chinese events

        Returns
        -------
        corpus: string, traditional Chinese events
        '''
        return st.convert(str(event))

    def dependency_parsing(self,corpus,pos):
        '''
        dependecy parse a corpus's sentence
        Parameters
        ----------
        corpus: double list , list of Chinese sentences(a sentence: [word,word,...])
        pos: double list , list of Chinese part-of-speeches(a sentence's pos: [pos, pos,...])

        Returns
        -------
        dep_corpus: double list , list of Chinese dependecy relation (a sentence's dependecncy: [(dep_index, dependecy),...])
        '''
        dep_corpus = []
        for sent,sent_pos in zip(corpus, pos) :
            dep_sents = []
            arcs = parser.parse(sent,sent_pos)
            for arc in arcs :
                dep_sents.append((arc.head - 1,arc.relation))
        
            dep_corpus.append(dep_sents) 
        return dep_corpus    
  

    def event_extraction(self,corpus, pos, dep, expand):
        '''
        extract events using corpus and its pos and dependency
        Parameters
        ----------
        corpus: double list , list of Chinese sentences(a sentence: [word,word,...])
        pos: double list , list of Chinese part-of-speeches(a sentence's pos: [pos, pos,...])
        dep: double list , list of Chinese dependecy relation (a sentence's dependecncy: [(dep_index, dependecy),...])
        expand: boolean, expand version or not
        Returns
        -------
        events: double list , list of corpus's events (an event: [word,word,...])
        '''
        events = []
        sent_idxs = []
        for j,(sent_word,sent_pos,sent_dep) in enumerate(zip(corpus,pos,dep)):
            sent = list(zip(sent_word, sent_pos, sent_dep))
            if sent == []:
                continue
            center_head_idx, head_list = self.find_sent_head(sent, expand)

            for head in head_list :

                head_idx = head[0] 
                predicate = head[1:]
                arg_bef,arg_aft = self.find_argument(sent,head_idx, expand)

                if expand:
                    if arg_bef != [] or arg_aft != []:
                        arg_bef = [arg for arg in arg_bef if arg not in predicate]
                        arg_aft = [arg for arg in arg_aft if arg not in predicate]
                        sig_event = arg_bef + predicate + arg_aft
                        events.append(sig_event)
                        sent_idxs.append(j)
                else:
                    if arg_bef == [] and arg_aft == [] :
                        continue
                
                    elif  arg_bef != [] and arg_aft == []:
                        for sig_arg_bef in arg_bef : 
                            sig_arg_bef = [arg for arg in sig_arg_bef if arg not in predicate]
                            sig_event = sig_arg_bef + predicate
                            events.append(sig_event)
                            sent_idxs.append(j)
                            
                    elif  arg_bef == [] and arg_aft != []:
                        for sig_arg_aft in arg_aft : 
                            sig_arg_aft = [arg for arg in sig_arg_aft if arg not in predicate]
                            sig_event = predicate + sig_arg_aft
                            events.append(sig_event)
                            sent_idxs.append(j)
            
                    else :
                        for sig_arg_bef in arg_bef : 
                            for sig_arg_aft in arg_aft:
                                sig_arg_bef = [arg for arg in sig_arg_bef if arg not in predicate]
                                sig_arg_aft = [arg for arg in sig_arg_aft if arg not in predicate]
                                sig_event = sig_arg_bef + predicate + sig_arg_aft
                                events.append(sig_event)
                                sent_idxs.append(j)
        return events, sent_idxs 

    def find_argument(self,sent,head_idx, expand):
        '''
        find arguments of a aentence

        Parameters:
        sent : list of tuples, [(word, pos, dep),(word, pos, dep),...]
        head_idx : int, index of head
        expand: boolean, expand version or not

        Return : 
        before & after arguments : list of words :[word,word,...,word]
        '''
        arg_bef = self.SOB_VOB(sent, head_idx, True, expand)
        arg_aft = self.SOB_VOB(sent, head_idx, False, expand)

        if arg_bef == []:
            arg_bef = self.ATR(sent, head_idx, True, expand)
        
        if arg_aft == []:
            arg_aft = self.CMO(sent, head_idx, False, expand)
        
        return arg_bef , arg_aft


    def idxtype(self,sent,idx,pos_dep, wh_type):
        '''
        to check if the word's pos/dep is the same as specified wh_type

        Parameters:
        sent : list of tuples, [(word, pos, dep),(word, pos, dep),...]
        idx : int, the position the required word is at
        pos_dep: boolean, whether you want to check pos. or dep.
        wh_type:string, type of pos./dep. you want to verify
        the available types can bee looked up in http://ltp.ai/docs/theory.html#id10

        Return :
        boolean, True if the word's pos/dep is the same as specified wh_type
        '''
        if pos_dep == 'pos':
            return sent[idx][1] == wh_type 
        elif pos_dep == 'dep' :
            return sent[idx][2][1] == wh_type      

    def find_sent_head(self, sent, expand):
        '''
        finding heads of a sentence

        Parameters:
        sent : list of tuples, [(word, pos, dep),(word, pos, dep),...]
        expand: boolean, expand version or not

        Return :
        int, center head index(word with dep. 'HED') ; double list, list of heads (a head: [index,word, word,...])
        '''
        negation = ['不','没','否','勿','无', '未', '非',' 毋']
        heads = []
        head = []
        
        dep_typs = [element[2][1] for element in sent]
        head_idx = dep_typs.index('HED')
        
        for idx in range(len(sent)) :
            head = []
            if expand:
                if self.idxtype(sent, idx, 'dep', 'HED'):
                    head.append(sent[idx][0])
                    if idx:
                        if self.idxtype(sent, idx - 1,'pos','p') :
                            head.insert(0,sent[idx - 1][0])
                        else :
                            try:
                                for i in range(idx - 1, -1, -1):
                                    if any((neg in sent[i][0]) for neg in negation):
                                        head.insert(0,sent[i][0])
                                    else :
                                        break 
                            except:
                                pass
            else:
                if self.idxtype(sent, idx, 'pos', 'v') or self.idxtype(sent, idx, 'dep', 'HED'):
                    head.append(sent[idx][0])
                    if idx:
                        if self.idxtype(sent, idx - 1,'pos','p') :
                            head.insert(0,sent[idx - 1][0])
                        else :
                            if idx:
                                for i in range(idx - 1, -1, -1):
                                    if any((neg in sent[i][0]) for neg in negation):
                                        head.insert(0,sent[i][0])   
                                    else:
                                        break               
            if head != [] :
                head.insert(0, idx)
                heads.append(head) 
            
        return head_idx, heads

    def SOB_VOB(self, sent, head_idx , before, expand):
        '''
        change
        expand:
            two layer:
            SBV/ FOB/ POB/ COO
            IOB/ DBL/ VOB/ FOB/ POB / COO 
        
            one layer:
            POB <- pos = p. <- HEAD
            HEAD -> pos = p. -> POB
    
        not expand:
            one layer:
            SBV/ FOB/ POB
            IOB/ DBL/ VOB/ FOB/ POB 
            POB <- pos = p. <- HEAD
            HEAD -> pos = p. -> POB
            COO replace coordinate word

        Parameters:
        sent : list of tuples, [(word, pos, dep),(word, pos, dep),...]
        head_idx : int, index of head
        expand: boolean, expand version or not
        before: boolean, if the argument we are looking for is before/after head, True if before
        
        Return :
        arg_idxes: list, argument's all indexes after enrichment
        '''
        arg_idxes_all = []
        args_all = []
        p_idx = []

        if before :
            dep_type = ['SBV', 'FOB', 'POB']
            if expand :
                dep_type += ['COO']
            
            start_idx = 0
            end_idx = head_idx

        else :
            dep_type = ['IOB', 'DBL', 'VOB','FOB','POB']
            if expand:
                dep_type += ['COO']
            
            start_idx = head_idx + 1
            end_idx = len(sent)
        
        arg_idxes = [idx for idx,element in enumerate(sent[start_idx: end_idx], start = start_idx) if (element[2][0] == head_idx) and (any(element[2][1] == dep_tp  for dep_tp in dep_type))]
    
        for idx,element in enumerate(sent[start_idx: end_idx], start = start_idx) :
            if element[2][0] == head_idx and self.idxtype(sent,idx,'pos', 'p'):
                p_idx.append(idx)
                for idx_inner,element in enumerate(sent[idx + 1: end_idx], start = idx + 1):
                    if element[2] == (idx, 'POB'):
                        arg_idxes.extend([idx, idx_inner])
                
        if expand:
            if arg_idxes:
                arg_idxes = sorted((set(arg_idxes) - set(p_idx)))
                arg_idxes = self.expand_so(arg_idxes, sent, start_idx, end_idx, dep_type)
                arg_idxes = self.enrich_outer(arg_idxes, sent, head_idx, before, expand)
                return [sent[idx][0] for idx in arg_idxes]
            else:
                return []
    
        else:
            if arg_idxes:
                arg_idxes_all.append(sorted(set(arg_idxes)))
                arg_idxes  = sorted((set(arg_idxes) - set(p_idx)))
            
                for idx in arg_idxes:
                    arg_idxes_coo = self.coo(arg_idxes, sent, head_idx, idx, before)
                    if arg_idxes_coo:
                        arg_idxes_coo.extend(p_idx)
                        arg_idxes_coo = sorted(set(arg_idxes_coo))
                        arg_idxes_all.append(arg_idxes_coo)

                for arg_idxes in arg_idxes_all:
                    arg_idxes = sorted((set(arg_idxes) - set(p_idx)))
                    arg_idxes = self.enrich_outer(arg_idxes, sent, head_idx, before, expand)
                    arg_idxes.extend(p_idx)
                    arg_idxes = sorted(set(arg_idxes))
                    args_all.append([sent[idx][0] for idx in arg_idxes])
                
        return args_all

    def ATR(self, sent, head_idx,before, expand):
        '''
        expand:
            if SUB none : infront all ATT,ATV  or (all (ATT, ATV),RAD) add
        not expand:
            if SUB none : infront ATT  or (ATT,RAD) add,  COO replace coordinate word

        Parameters:
        sent : list of tuples, [(word, pos, dep),(word, pos, dep),...]
        head_idx : int, index of head
        expand: boolean, expand version or not
        before: boolean, if the argument we are looking for is before/after head, True if before
        
        Return :
        arg_idxes: list, argument's all indexes after enrichment
        '''
        arg_idxes = []
        arg_idxes_all = []
        args_all = []
    
        dep_type = ['ATT']
        if expand:
            dep_type += ['ADV']
    
        start_idx = 0
        end_idx = head_idx
    
        if expand:
            try:
                for idx in range(head_idx - 1, -1, -1):
                    word,pos,(dep_idx,dep) = sent[idx]
        
                    if any(dep == dep_tp for dep_tp in dep_type):
                        arg_idxes.append(idx)
                    else:
                        break
                    
                if sent[head_idx - 1][2][1] == 'RAD' :
                    for idx in range(head_idx - 2, -1, -1):
                        word,pos,(dep_idx,dep) = sent[idx]
        
                        if any(dep == dep_tp for dep_tp in dep_type):
                            arg_idxes.append(idx)
                        else:
                            break
                    if arg_idxes:
                        arg_idxes.append(head_idx - 1)
            except:
                pass
                    
            if arg_idxes:
                arg_idxes = self.expand_so(arg_idxes, sent, start_idx, end_idx, dep_type)
                arg_idxes = self.enrich_outer(arg_idxes, sent, head_idx, before, expand)
                return [sent[idx][0] for idx in arg_idxes]
            else :
                return []
        
        else:
            if head_idx - 1:
                dep = sent[head_idx - 1][2][1]
                if dep == 'ATT' or dep == 'ADV' or dep == 'RAD': arg_idxes = [head_idx - 1]
        
            if arg_idxes:
                arg_idxes_all.append(arg_idxes)
            
                for idx in arg_idxes:
                    arg_idxes_coo = self.coo(arg_idxes, sent, head_idx, idx, before)
                    if arg_idxes_coo:
                        arg_idxes_all.append(arg_idxes_coo)
   
                for arg_idxes in arg_idxes_all:
                    arg_idxes = self.enrich_outer(arg_idxes, sent, head_idx, before, expand)
                    args_all.append([sent[idx][0] for idx in arg_idxes])
                
            return args_all

    def CMO(self, sent, head_idx,before, expand):
        '''
        expand:
            if OBJ none : after all (CMP or COO) add
        not expand:
            if OBJ none : after CMP or COO add,  COO replace coordinate word

        Parameters:
        sent : list of tuples, [(word, pos, dep),(word, pos, dep),...]
        head_idx : int, index of head
        expand: boolean, expand version or not
        before: boolean, if the argument we are looking for is before/after head, True if before
        
        Return :
        arg_idxes: list, argument's all indexes after enrichment
        '''
        arg_idxes = []
        arg_idxes_all = []
        args_all = []
    
        dep_type = ['CMP','COO']
    
        start_idx = head_idx + 1
        end_idx = len(sent)

        if expand:
            try:
                for idx in range(head_idx + 1, len(sent)):
                    word,pos,(dep_idx,dep) = sent[idx]

                    if any(dep == dep_tp for dep_tp in dep_type):
                        arg_idxes.append(idx)
                    else:
                        break
            except:
                pass
                    
            if arg_idxes:
                arg_idxes = self.expand_so(arg_idxes, sent, start_idx, end_idx, dep_type)
                arg_idxes = self.enrich_outer(arg_idxes, sent, head_idx, before, expand)
                return [sent[idx][0] for idx in arg_idxes]
            else:
                return []
        
        else:
            if head_idx + 1 < len(sent):
                dep = sent[start_idx][2][1]
                if any(dep == dep_tp for dep_tp in dep_type): arg_idxes = [start_idx]
            
            if arg_idxes:
                arg_idxes_all.append(arg_idxes)
            
                for idx in arg_idxes:
                    arg_idxes_coo = self.coo(arg_idxes, sent, head_idx, idx, before)
                    if arg_idxes_coo:
                        arg_idxes_all.append(arg_idxes_coo)
   
                for arg_idxes in arg_idxes_all:
                    arg_idxes = self.enrich_outer(arg_idxes, sent, head_idx, before, expand)
                    args_all.append([sent[idx][0] for idx in arg_idxes])
                
            return args_all

    def expand_so(self, arg_idxes, sent, start_idx, end_idx, dep_type):
        '''
        expand before/ after arguments by finding their arguments too

        Parameters
        ----------
        arg_idxes : list, argument's all indexes 
        sent : list, tokenized sentence
        head_idx : int, index of sentence's head
        start_idx : int, the position of sentence to start searching
        end_idx : int, the position of sentence to end searching
        dep_type : list of strings, candidate dependency types 

        Returns
        -------
        arg_idxes : list, argument's all indexes after enrichment
        '''
        for search_idx in arg_idxes:
            new_arg_idxes = [idx for idx,element in enumerate(sent[start_idx: end_idx], start = start_idx) if (element[2][0] == search_idx) and (any(element[2][1] == dep_tp  for dep_tp in dep_type))]
            arg_idxes.extend(new_arg_idxes)
        return sorted(set(arg_idxes))

    def enrich_outer(self, arg_idxes, sent, head_idx, before, expand):
        '''
        enrich original argument's all words

        Parameters
        ----------
        arg_idxes : list, argument's all indexes 
        sent : list, tokenized sentence
        head_idx : int, index of sentence's head
        before: boolean, check if the argument is before or after the head
        expand: boolean, version of events

        Returns
        -------
        arg_idxes : list, argument's all indexes after enrichment
        '''
        for idx in arg_idxes:
            arg_idxes = self.enrich(arg_idxes, sent, head_idx, idx, before,expand)
        return sorted(set(arg_idxes))

    def enrich(self, arg_idxes ,sent, head_idx, arg_idx, before, expand):
        '''
        enrich original argument's single word

        Parameters
        ----------
        arg_idxes : list, argument's all indexes 
        sent : list, tokenized sentence
        head_idx : int, index of sentence's head
        arg_idx : int, index of the word in the argument needed to be enriched
        before: boolean, check if the argument is before or after the head
        expand: boolean, version of events

        Returns
        -------
        arg_idxes : list, argument's all indexes after enrichment
        '''
        if expand:
            arg_idxes_atr = self.atr(arg_idxes, sent, head_idx, arg_idx, before, expand)
        else:
            arg_idxes_atr = []

        arg_idxes_qm = self.qm(arg_idxes, sent, head_idx, arg_idx, before, expand)
        arg_idxes_infp = self.inf_p(arg_idxes, sent, head_idx, arg_idx, before, expand)
        arg_idxes_oneword = self.one_word(arg_idxes, sent, head_idx, arg_idx, before, expand)

        arg_idxes = sorted(set(arg_idxes_atr + arg_idxes_qm + arg_idxes_infp + arg_idxes_oneword))
    
        return arg_idxes

    def coo(self, arg_idxes, sent, head_idx, arg_idx, before):
        '''
        change
        not expand:
            find the argument's word's coordinate words and replace it, therefore produce a new argument

        Parameters
        ----------
        arg_idxes : list, argument's all indexes 
        sent : list, tokenized sentence
        head_idx : int, index of sentence's head
        arg_idx : int, index of the word in the argument needed to be enriched
        before: boolean, check if the argument is before or after the head

        Returns
        -------
        arg_idxes : list, argument's all indexes after enrichment
        '''
        coo_idx = []
        if before and arg_idx < (len(sent) - 1):
            coo_idx = [idx for idx,element in enumerate(sent[arg_idx + 1 : head_idx], start = arg_idx + 1) if element[2] == (arg_idx,'COO')]       
        elif arg_idx < (len(sent) - 1):
            coo_idx = [idx for idx,element in enumerate(sent[arg_idx + 1 :], start = arg_idx + 1) if element[2] == (arg_idx,'COO')]
        if coo_idx:
            arg_idxes_coo.remove(arg_idx)
            arg_idxes_coo.append(coo_idx[0])
            return sorted(set(arg_idxes_coo))
        else:
            return False

    def atr(self, arg_idxes, sent, head_idx, arg_idx, before, expand):
        '''
        expand:
            if SUB or OBJ infront word's dep. is 'ATT' or 'ADV' or '(ATT or ADV) + RAD' : word add
        Parameters
        ----------
        arg_idxes : list, argument's all indexes 
        sent : list, tokenized sentence
        head_idx : int, index of sentence's head
        arg_idx : int, index of the word in the argument needed to be enriched
        before: boolean, check if the argument is before or after the head
        expand: boolean, version of events

        Returns
        -------
        arg_idxes : list, argument's all indexes after enrichment
        '''
        arg_idxes_cp = arg_idxes.copy()
        if arg_idx:
            idx = arg_idx - 1
        
            if (self.idxtype(sent, idx, 'dep', 'ATT') or self.idxtype(sent, idx, 'dep', 'ADV')) and idx != head_idx:
                arg_idxes_cp.append(idx)
            
            elif self.idxtype(sent, idx, 'dep','RAD') and idx != head_idx:
                if idx:
                    idx -= 1
                    if self.idxtype(sent,idx,'dep','ATT') or self.idxtype(sent,idx,'dep','ADV'):
                        arg_idxes_cp.append(idx)
                        arg_idxes_cp.append(idx + 1)
                    
        return sorted(set(arg_idxes_cp))

    def qm(self, arg_idxes, sent, head_idx, arg_idx, before, expand):
        '''
        expand:
            if SUB or OBJ's infront word's pos = m. and itself's pos = q.(recursively) : word add
        not expand:
            if SUB or OBJ's infront word's pos = m. and itself's pos = q.: word add

        Parameters
        ----------
        arg_idxes : list, argument's all indexes 
        sent : list, tokenized sentence
        head_idx : int, index of sentence's head
        arg_idx : int, index of the word in the argument needed to be enriched
        before: boolean, check if the argument is before or after the head
        expand: boolean, version of events

        Returns
        -------
        arg_idxes : list, argument's all indexes after enrichment
        '''
        arg_idxes_cp = arg_idxes.copy()
        if not self.idxtype(sent, arg_idx,'pos','q') and not self.idxtype(sent, arg_idx,'pos','m'):
            return sorted(set(arg_idxes))
    
        if expand :
            if before :
                ite_seq = range(arg_idx - 1, -1,-1)
            else :
                ite_seq = range(arg_idx - 1, head_idx, -1) 
            try:
                for idx in ite_seq:
                    if (self.idxtype(sent, idx,'pos','m') or self.idxtype(sent, idx,'pos','q')) and idx != head_idx:
                        arg_idxes_cp.append(idx)
                    else:
                        break
            except:
                pass
        
        else:
            if arg_idx:
                if (self.idxtype(sent, arg_idx - 1,'pos','m') or self.idxtype(sent, arg_idx - 1,'pos','q')) and arg_idx - 1 != head_idx :
                    arg_idxes_cp.append(arg_idx - 1)
        
        return sorted(set(arg_idxes_cp))

    def inf_p(self, arg_idxes, sent, head_idx, arg_idx, before, expand):
        '''
        expand:
            if SUB or OBJ infront word's pos = p. (recursively): word add
        not expand:
            if SUB or OBJ infront word's pos = p. : word add'
        Parameters
        ----------
        arg_idxes : list, argument's all indexes 
        sent : list, tokenized sentence
        head_idx : int, index of sentence's head
        arg_idx : int, index of the word in the argument needed to be enriched
        before: boolean, check if the argument is before or after the head
        expand: boolean, version of events

        Returns
        -------
        arg_idxes : list, argument's all indexes after enrichment
        '''
        arg_idxes_cp = arg_idxes.copy()
        if expand:
            if before :
                ite_seq = range(arg_idx - 1, -1,-1)
            else :
                ite_seq = range(arg_idx - 1, head_idx, -1)

            try:
                for idx in ite_seq:
                    if self.idxtype(sent,idx,'pos','p') and idx != head_idx:
                        arg_idxes_cp.append(idx)
                    else:
                        break
            except:
                pass      
        else:
            if arg_idx:
                if self.idxtype(sent, arg_idx - 1, 'pos', 'p') and arg_idx - 1 != head_idx:
                    arg_idxes_cp.append(arg_idx - 1)
        return sorted(set(arg_idxes_cp))

    def one_word(self, arg_idxes, sent, head_idx, arg_idx, before, expand):
        '''
        expand & not expand:
            if SUB or OBJ one word and pos. is not p, q, and m :one infront word add

        Parameters
        ----------
        arg_idxes : list, argument's all indexes 
        sent : list, tokenized sentence
        head_idx : int, index of sentence's head
        arg_idx : int, index of the word in the argument needed to be enriched
        before: boolean, check if the argument is before or after the head
        expand: boolean, version of events

        Returns
        -------
        arg_idxes : list, argument's all indexes after enrichment
        ''' 
        arg_idxes_cp = arg_idxes.copy()
        word = sent[arg_idx][0]
        if len(word) > 1 :
            return sorted(set(arg_idxes))
        
        elif arg_idx and arg_idx - 1 != head_idx:
            arg_idxes_cp.append(arg_idx - 1)
            
        return sorted(set(arg_idxes_cp))