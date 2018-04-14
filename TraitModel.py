from tqdm import tqdm
import pandas as pd
import numpy as np
import re
from re import findall as fa
import sqlite3
import pymorphy2
import math
import importlib
import pickle

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier 
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from scipy.stats.stats import pearsonr as corr

from nltk.tokenize import word_tokenize

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, confusion_matrix



lib = {
    'allpos': ['PRED', 'None', 'PRTS', 'ADJF', 'INFN', 
         'PRTF', 'NOUN', 'ADVB', 'VERB', 'NPRO', 
         'NUMR', 'CONJ', 'ADJS', 'PRCL', 'PREP', 'COMP', 'INTJ'],
    'pos': ['ADJF', 'NOUN', 'ADVB', 'VERB', 'CONJ', 'PREP', 'INTJ', 'None'],
    'uncert' : ['наверное?', 'может[\s-]?быть', 'кажеть?ся', 
           'видимо', 'возможно', 'по[\s-]?видимому', 
           'вероятно', 'должно[\s-]?быть','пожалуй', 'как[\s-]?видно'],
    'cert' : ['очевидно','конечно','точно','совершенно',
         'не\s?сомненно','разумееть?ся', 
         'по[\s-]?любому','сто[\s-]?пудово?'],
    'quan' : ['вс[её]x?','всегда','ни-?когда', 'постоянно', 
         'ник(?:то|ого|ому|ем)', 
         'кажд(?:ый|ая|ой|ому?|ое|ого|ую|ые|ою|ыми?|ых)',
         'всяк(?:ий|ая|ое|ого|ую|ому?|ой|ою|ими?|их|ие)',
         'люб(?:ой|ая|ое|ого|ому?|ую|ой|ыми?|ых|ые)'],
    'imper' : ['долж(?:ен|на|ны|но)', 'обязан(?:а|ы|о|)', 
          'надо\W', 'нуж(?:но|ен|на|ны)', 
          'требуеть?ся', 'необходим(?:а|ы|о|)\W'],
    'racio' : ['по\s?этому', 'по\s?тому,?\s?что', 'следовательно', 
          'из[\s-]?за\s?того,?\s?что', 'из[\s-]?за\s?этого', 
          'по\s?причине', 'в\s?следстви[ие]', 'так\s?как', 'т\.?к\.?',
          'поскольк[оу]', 'чтобы'],
    'dimin' : ['\w+[ое]ньк(?:ая|ий|ое|ие|ую|ого|ому|ой|ими|а|о|у|е)', 
          '\w+очек\s', '\w+[ие]к(?:ами?|ов|у|а|е|и|)\s'],
    'extrem' : ['че?резвычайно', 'слишком', 'чере[cз]чур', 'ужасно',
           'безумно', 'крайне', 'предельно',
           'исключительно', 'невероятно', 'в\s?(?:наи|)вы[сш]шей (?:степени|мере)'],
    'like' : ['люблю', '[оa]б[оa]жаю','восхища[ею]т',
        'в\s?восторге', 'нрав[ия]ть?ся'],
    'dislike' : ['бес[ия]т', 'ненавижу', 'терпеть\s?не\s?могу', 
           'раздража[ею]т', 'зл[ия]т', 'выбешива[ею]т', ],
    'polite' : ['пожалуй?ст[ао]', 'пожал[ао]ст[ао]', 
                'с?пасиб[оа]?', 'благ[оа]д[ао]рю'],
    'obscene' : ['\sбля\s', '\sбля[дт]ь\s', '\sсук(?:ами?|ax|у|а|е|и|)\s', 
           '\sху(?:ями?|ем|ях|ю|е|я|й)\s', '\sна\s?хуй\s', 
           '\w*(?:подъ?|за|на|вы|по|при|от|у|)еб[лa]?\w*',
           'мудак\w*', 'мудил\w+\s','пид[оa]р\w*', 
           'пид[ао]рас\w*\s'],
    'slang' : ['\s\w*хайп\w*\s', 'хейтер\w*','\sчилить','\sизи\w*\s','зашквар\w*',
         '\sжиз\w+','\sкун\w*','\sтянк?\w*','\sлойс\w?', '\sсорян\s'],
}


def ct(x, co=0, steep=0, ec50=0.5, level='max', adb=True):
    def carryover(x, l=0):
        co = []
        co.append((1-l)*x[0])
        for i in range(1,len(x)):
            co.append((1-l)*x[i] + l*co[i-1])
        return co

    def adbudg(x, steep, ec50, level):
        cap = max(x) if level == 'max' else level
        adb = []
        for i in x:
            adb.append(0 if i == 0 else cap/(1 + (i/(cap*ec50))**(-steep))) 
        return adb

    def logcurve(x, steep, ec50, level):
        cap = max(x) if level == 'max' else level*max(x)
        crv = []
        for i in x:
            crv.append(cap/(1+math.exp((-steep)*(i/cap-ec50))) - cap/(1+math.exp(steep*ec50)))
        return crv
      
    ct = x
    if steep > 0:
        ct = adbudg(x, steep, ec50, level) if adb else logcurve(x, steep, ec50, level)
    ct = carryover(ct, co) if co > 0 else ct
    return ct

def cleanse(s):
    rgxp = '[\`\)\(\|©~^<>/\'\"\«№#$&\*.,;=+?!\—_@:\]\[%\{\}\\n]'
    return re.sub(' +', ' ', re.sub(rgxp, ' ', s.lower()))

def set_groups(x, dev=1, M=50, SD=10):
    if x > M+dev*SD:
        return 'high'
    elif x < M-dev*SD:
        return 'low'
    else:
        return 'average'
    
def mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def extract_features(text, morph=pymorphy2.MorphAnalyzer(), 
                 pos_types=lib['pos'], 
                 uncert=lib['uncert'],
                 cert=lib['cert'], 
                 quan=lib['quan'], 
                 imper=lib['imper'], 
                 racio=lib['racio'], 
                 dimin=lib['dimin'], 
                 extrem=lib['extrem'],
                 like=lib['like'],
                 dislike=lib['dislike'], 
                 polite=lib['polite'], 
                 obscene=lib['obscene'], 
                 slang=lib['slang']):
    
    from re import findall as fa
    #length in chars and words
    len_char = len(text)
    len_word = len(text.split())
    len_sent = len(fa('[^\.\!\?]+[\.\!\?]', text))
    len_sent = len_sent if len_sent else 1
    pun = fa('[\.+,!\?:-]',text)
    n_pun = len(pun)
    braсket_list = fa('[\(\)]',text)
      
    #POS & grammem    
    def parse_text(text, morph=morph):
        tokens = cleanse(text).split()
        return [morph.parse(t) for t in tokens]
    
    parsed_text = parse_text(text)
    pos_list = [str(p[0].tag.POS) for p in parsed_text]
    n_nouns = len([t for t in pos_list if t=='NOUN'])
    n_verbs = len([t for t in pos_list if t=='VERB'])
    n_ad = len([t for t in pos_list if t in ['ADJF','ADVB']])
    anim_list = [str(p[0].tag.animacy) for p in parsed_text]
    pers_list = [str(p[0].tag.person) for p in parsed_text]
    tns_list = [str(p[0].tag.tense) for p in parsed_text]
    asp_list = [str(p[0].tag.aspect) for p in parsed_text]
      
    r = lambda x: round(x, 4)
    d = lambda x, y: x / y if y else 0.0
    
    features = {
        #surface features
        'len_char': len_char, 
        'len_word': len_word,
        'len_sent': len_sent,
        'm_len_word': r(len_char / len_word),
        'm_len_sent': r(len_word / len_sent),
        #punctuation
        'p_pun': r(len(pun) / len_char),
        'p_dot': r(d(len([i for i in pun if i=='.']), len(pun))),
        'p_qm': r(d(len([i for i in pun if i=='?']), len(pun))),
        'p_excl': r(d(len([i for i in pun if i=='!']), len(pun))),
        'p_comma': r(d(len([i for i in pun if i==',']), len(pun))),
        'p_brkt': r(len(braсket_list) / len_char),
        'p_brkt_up': r(d(len([i for i in braсket_list if i==')']), len(braсket_list))),
        #POS form
        'pos_form': ' '.join(pos_list),
        'pos_richness': len(set(pos_list)),
        #grammem features
        'p_anim': r(d(len([t for t in anim_list if t=='anim']), n_nouns)),
        'p_1per': r(d(len([t for t in pers_list if t=='1per']), n_verbs)),
        'p_3per': r(d(len([t for t in pers_list if t=='3per']), n_verbs)),
        'p_past': r(d(len([t for t in tns_list if t=='past']), n_verbs)),
        'p_fut': r(d(len([t for t in tns_list if t=='futr']), n_verbs)),
        'p_pres': r(d(len([t for t in tns_list if t=='pres']), n_verbs)),
        'p_perf': r(d(len([t for t in asp_list if t=='perf']), n_verbs)),
        'p_conj': r(d(len(fa('\sбы?\s',text)), n_verbs)),
        #lexical features
        'p_uncert': r(len(fa('|'.join(uncert), text.lower())) / len_word),
        'p_cert': r(len(fa('|'.join(cert), text.lower())) / len_word),
        'p_quan': r(len(fa('|'.join(quan), text.lower())) / len_word),
        'p_imper': r(len(fa('|'.join(imper), text.lower())) / len_word),
        'p_racio': r(len(fa('|'.join(racio), text.lower())) / len_word),
        'p_dimin': r(len(fa('|'.join(dimin), text.lower())) / len_word),    
        'p_extrem': r(len(fa('|'.join(extrem), text.lower())) / len_word), 
        'p_like': r(len(fa('|'.join(like), text.lower())) / len_word),    
        'p_dislike': r(len(fa('|'.join(dislike), text.lower())) / len_word),  
        'p_polite': r(len(fa('|'.join(polite), text.lower())) / len_word),  
        'p_obscene': r(len(fa('|'.join(obscene), text.lower())) / len_word), 
        'p_slang': r(len(fa('|'.join(slang), text.lower())) / len_word)
    }
    
    for f in pos_types:
        features['p_'+f] = r(len([t for t in pos_list if t==f])/len(pos_list))
        
    return features


class TraitModel():
    def __init__(self, xname, traits, word_vectorizer, pos_vectorizer, 
                 library, test_size, morph, classifier, classifier_params={}, 
                 curves_params={'co':0, 'steep':0, 'ec50':0.5}):
        self.xname = xname
        self.traits = traits
        self.wv = word_vectorizer
        self.posv = pos_vectorizer
        self.lib = library
        self.test_size = test_size
        self.morph = morph
        self.models = {}
        self.quality = {'train':{}, 'test':{}}
        self.cl = classifier
        self.cl_params = classifier_params
        self.ct_params = curves_params
    
    
    def fit(self, data, mtype='n', summary=True):
        self.mtype = mtype       
        #extract features
        df_feat = pd.DataFrame.from_records(list(data[self.xname].apply(extract_features, 
                                                                        morph=self.morph)))
        df_feat.index = data.index
        data = pd.concat([data, df_feat], axis=1, join='inner')
        feat_names = list(extract_features('ы', morph=self.morph).keys())
        feat_names.remove('pos_form')
        #apply curve transformation
        for f in feat_names:
            data[f] = ct(data[f], **self.ct_params)     
        #clean before vectorization
        data[self.xname] = data[self.xname].apply(cleanse)
        #train-test split 
        train, test = train_test_split(data, test_size=self.test_size, random_state=42)
        #vectorize
        train_w_vec = self.wv.fit_transform(train.loc[:,'text']) #words tf:idf
        test_w_vec = self.wv.transform(test.loc[:,'text'])
        train_p_vec = self.posv.fit_transform(train.loc[:,'pos_form']) #pos tf:idf
        test_p_vec = self.posv.transform(test.loc[:,'pos_form'])
        X_train = np.hstack((train_w_vec.todense(), 
                             train_p_vec.todense(), 
                             train.loc[:,feat_names]))
        X_test = np.hstack((test_w_vec.todense(), 
                            test_p_vec.todense(), 
                            test.loc[:,feat_names]))
        
        self.fitted_features = self.wv.get_feature_names() \
                             + self.posv.get_feature_names() \
                             + feat_names 
        self.data = data
        self.train = train
        self.test = test
        self.X_train = X_train
        self.X_test = X_test
        self.feat_names = feat_names
        
        # build feature models
        def build_model(X_train, X_test, y_train, y_test, model):
            model.fit(X_train, y_train)
            return model
        
        for trait in self.traits:
            lm = self.cl(**self.cl_params)
            trait = trait+'_nom' if self.mtype == 'n' else trait          
            self.models[trait] = build_model(self.X_train, self.X_test, 
                                             self.train.loc[:,trait], self.test.loc[:,trait], 
                                             model=lm)
        
        if summary: self.summary(confusion=False, verbose=False)
    
    
    def summary(self, correlations=0.1, vec_tokens=15, coefs = 10,
                all_traits=True, clr=True, confusion=True, verbose=True):
        pr = (lambda x: print(x)) if verbose else (lambda x: None)
        def title(name, m='=', l=50, up=True):
            n = name.upper() if up else name
            return '\n{}\n{}\n{}'.format(m*l, n, m*l)
        
        if correlations > 0:
            pr(title('CORRELATIONS'))
            for trait in self.traits:
                pr(title(trait, m='-', l=20, up=False))
                for feat in self.feat_names:
                    cor = corr(self.data.loc[:,trait], self.data.loc[:,feat])
                    if abs(cor[0]) > correlations:
                        pr('{} | {} : r = {:.2}'.format(feat, trait, cor[0], cor[1]))
                        
        def vect_summary(vectorizer, name, show_tokens):
            pr(title(name))
            pr('\nIncluded tokens ({})'.format(len(vectorizer.get_feature_names())))
            pr(np.array(vectorizer.get_feature_names())\
                  [np.random.randint(0, len(vectorizer.get_feature_names()), show_tokens)])
            pr('\nExcluded tokens ({})'.format(len(vectorizer.stop_words_)))
            pr(np.array(list(vectorizer.stop_words_))\
                  [np.random.randint(0, len(vectorizer.stop_words_), show_tokens)]) 
            
        if vec_tokens > 0:
            vect_summary(self.wv, 'words', vec_tokens)
            vect_summary(self.posv, 'pos tags', vec_tokens)            
        
        pr(title('prediction quality'))
        if self.mtype == 'n' and all_traits:
            for trait in self.traits:
                trait = trait+'_nom'
                pr(title(trait, m='-', l=20, up=False))
                model = self.models[trait]
                y_train = self.train.loc[:,trait]
                y_test = self.test.loc[:,trait]
                y_train_pred = model.predict(self.X_train)
                y_test_pred = model.predict(self.X_test)
                self.quality['train'][trait+'_acc'] = accuracy_score(y_train, y_train_pred)
                self.quality['test'][trait+'_acc'] = accuracy_score(y_test, y_test_pred)
                pr('\nAccuracy on train: {:.2%}'.format(accuracy_score(y_train, y_train_pred)))
                if clr: pr(classification_report(y_train, y_train_pred))
                pr('Accuracy on test: {:.2%}'.format(accuracy_score(y_test, y_test_pred)))
                if clr: pr(classification_report(y_test, y_test_pred))
                if confusion:
                    labels = y_train.unique()
                    sns.set_context("notebook")
                    plt.figure(figsize=(4,3))
                    sns.heatmap(data=confusion_matrix(y_test, y_test_pred, labels = labels), 
                                annot=True, fmt="d", cbar=False, 
                                xticklabels=labels, yticklabels=labels, cmap='viridis')
                    plt.title("Confusion matrix")
                    plt.xlabel('True')
                    plt.ylabel('Predicted')
                    plt.title("Confusion matrix for "+y_train.name, fontsize=12, fontweight='bold');
                    plt.show()
        
        if self.mtype == 'c' and all_traits:       
            for trait in self.traits:
                pr(title(trait, m='-', l=20, up=False))
                model = self.models[trait]
                y_train = self.train.loc[:,trait]
                y_test = self.test.loc[:,trait]
                y_train_pred = model.predict(self.X_train)
                y_test_pred = model.predict(self.X_test)
                self.quality['train'][trait+'_R2'] = r2_score(y_train, y_train_pred)
                self.quality['test'][trait+'_R2'] = r2_score(y_test, y_test_pred)
                pr('MAPE on train: {:.2f}%'.format(mape(y_train, y_train_pred)))
                pr('R2 on train: {:.3f}'.format(r2_score(y_train, y_train_pred)))
                pr('\nMAPE on test: {:.2f}%'.format(mape(y_test, y_test_pred)))
                pr('R2 on train: {:.3f}'.format(r2_score(y_test, y_test_pred)))
        
        pr(title('Mean quality', m='-', l=20, up=False))
        metric = 'R2' if self.mtype == 'c' else 'Accuracy'
        for k,v in self.quality.items():
            pr('Mean {} on {}: {:.3}'.format(metric, k, sum(v.values())/len(v.values())))
            
        if coefs and str(type(list(self.models.values())[0])) == \
        "<class 'sklearn.linear_model.logistic.LogisticRegression'>":
            pr(title('coefs'))
            for trait in self.traits: 
                trait = trait + '_nom' if self.mtype == 'n' else trait
                pr(title(trait, m='-', l=20, up=False))
                for i, level in enumerate(self.models[trait].classes_):
                    pr('\n'+level.upper())
                    features = self.fitted_features
                    coefs_ = self.models[trait].coef_.tolist()[i]
                    coefdf = pd.DataFrame({'Feature' : features, 
                                           'Coefficient' : coefs_})
                    coefdf = coefdf.sort_values(['Coefficient', 'Feature'], ascending=[0, 1])
                    pr(coefdf.head(coefs))
    
    
    def export(self, path='models/', fname='model'):
        with open(path+fname+'.pickle', 'wb') as f:
            pickle.dump(self, f)
            
            
    def predict(self, text, prob=False, stimulus=True):
        feats = extract_features(text, morph=self.morph)
        featvec = np.array([feats[f] for f in self.feat_names])
        X = np.hstack((self.wv.transform([text]).todense(), 
                       self.posv.transform([feats['pos_form']]).todense(), 
                       np.matrix(featvec)))
        if prob:
            predictions = {}
            for trait in self.traits:
                trait = trait + '_nom' if self.mtype == 'n' else trait
                pred = self.models[trait].predict_proba(X)
                predictions[trait] = {clss: pred[0][i] for i, clss \
                                      in enumerate(self.models[trait].classes_)}
        else:
            predictions = {k:v.predict(X) for k, v in self.models.items()}
        if stimulus: 
            predictions['X'] = X 
        return predictions


def plot_traits(model, text, strict=True, mqw = 0.5, corrw = 0.3, textlen=700):
    values = {'low': 1, 'average': 2, 'high': 3}
    sd = 1
    traits = model.traits
    trait_labels = ['Extraversion', 'Agreeableness', 'Conscientiousness', 
                    'Emotionality', 'Openness to Experience', 'Honesty-Humility']

    pred = model.predict(text, prob=False, stimulus=False)
    predp = model.predict(text, prob=True, stimulus=False)
    yval = [values[pred[trait+'_nom'][0]] - \
            predp[trait+'_nom']['low']*corrw + \
            predp[trait+'_nom']['high']*corrw \
            for trait in traits]
    xval = np.arange(len(traits))
    limits = [sd*(1-predp[trait+'_nom'][pred[trait+'_nom'][0]]+0.25) for trait in traits]

    textlen_penalty = 1 if len(text) > textlen else len(text) / textlen
    if strict:
        conf = np.mean(list(model.quality['test'].values())) * \
            np.mean([max(v.values()) for v in predp.values()]) * textlen_penalty
    else:
        conf = (np.mean(list(model.quality['test'].values()))*(mqw) + \
                np.mean([max(v.values()) for v in predp.values()])*(1-mqw)) * textlen_penalty

    fig = plt.figure(figsize=(7,6))
    plt.xticks(xval, trait_labels, rotation=-45)
    plt.yticks(sorted(values.values()), ('Low', 'Average', 'High'))
    colors = ['tab:orange', 'tab:green', 'tab:blue', 'tab:cyan', 'tab:purple', 'tab:olive']
    plt.xlabel('Trait', fontsize=14)
    plt.ylabel('Level', fontsize=14)
    for i, c in enumerate(colors):  
        plt.errorbar(xval[i], yval[i], yerr=limits[i], 
                     uplims=True, lolims=True, 
                     fmt='o', markersize='32',ecolor=c, 
                     markerfacecolor=c, elinewidth=2, capsize=4)
    plt.ylim(min(min(yval), min(values.values()))-max(limits)*1.5, 
             max(max(values.values()), max(yval))+max(limits)*1.5)
    plt.xlim(min(xval)-0.5, max(xval)+0.5)
    plt.title(('Predicted profile (model confidence: {:.1%})'.format(conf)), 
              fontsize=16, fontweight='bold');
    fig.patch.set_facecolor('0.2')
    plt.show();