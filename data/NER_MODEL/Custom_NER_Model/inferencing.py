from __future__ import unicode_literals, print_function
import plac
import spacy
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from sklearn.metrics import accuracy_score


nlp = spacy.load('Custom_NER_Model4')
test = convert_doccano_fomart_to_spacy('file(9).json1')

def test_spacy():
    #test the model and evaluate it
    examples = test
    tp=0
    tr=0
    tf=0

    ta=0
    c=0        
    for text,annot in examples:
      # print(text)
        if "NET&SYN TALENT REPORT" in text:
          REUSE_TYPE = "NET&SYN TALENT REPORT"
        elif "ISCI USAGE BY CODE" in text:
          REUSE_TYPE = "ISCI USAGE BY CODE"
        else:
          REUSE_TYPE = "NONE"
        if "NT:TALHRPT" in text:
          REPORT_FORMAT = "NT:TALHRPT"
        else:
          REPORT_FORMAT = "NONE"

        doc_to_test=nlp(text)
        d={}
        for ent in doc_to_test.ents:
            d[ent.label_]=[]
        for ent in doc_to_test.ents:
            d[ent.label_].append(ent.text)
        d['REPORT_FORMAT'] = [REPORT_FORMAT]
        d['REUSE_TYPE'] = [REUSE_TYPE]
        with open("test"+str(c)+".json","w") as outputfile:
          json.dump(d , outputfile)
        d={}
        for ent in doc_to_test.ents:
            d[ent.label_]=[0,0,0,0,0,0,0]
        for ent in doc_to_test.ents:
            doc_gold_text= nlp.make_doc(text)
            gold = GoldParse(doc_gold_text, entities=annot.get("entities"))
            y_true = [ent.label_ if ent.label_ in x else 'Not '+ent.label_ for x in gold.ner]
            y_pred = [x.ent_type_ if x.ent_type_ ==ent.label_ else 'Not '+ent.label_ for x in doc_to_test]  
            if(d[ent.label_][0]==0):
                (p,r,f,s)= precision_recall_fscore_support(y_true,y_pred,average='weighted')
                a=accuracy_score(y_true,y_pred)
                d[ent.label_][0]=1
                d[ent.label_][1]+=p
                d[ent.label_][2]+=r
                d[ent.label_][3]+=f
                d[ent.label_][4]+=a
                d[ent.label_][5]+=1
        c+=1
    for i in d:
        print("\n For Entity "+i+"\n")
        print("Accuracy : "+str((d[i][4]/d[i][5])*100)+"%")
        print("Precision : "+str(d[i][1]/d[i][5]))
        print("Recall : "+str(d[i][2]/d[i][5]))
        print("F-score : "+str(d[i][3]/d[i][5]))
test_spacy()