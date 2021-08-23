## preparation and structuring Dataset 

### resume data
### Extracting data for hiring a developer 

import numpy as np
import spacy
import tika
from spacy.matcher import Matcher
from tika import parser


class Resume_Extractor:
  doc_list=[]

  def __init__(self,resumelist):
    nlp = spacy.load("en_core_web_sm")
    
    for resume in resumelist:
      raw = parser.from_file(resume)
      # print(raw['content'])
    
      self.content=raw['content']
      self.doc=nlp(self.content)

      Resume_Extractor.doc_list.append(self.doc)

  def __call__(self,matcher):
    matches=[]
    for n_doc in Resume_Extractor.doc_list:
      match=matcher(n_doc)
      matches.append(match)
    return matches,Resume_Extractor.doc_list


class Feature_Matrix():
  def __init__(self,n_resumes,n_features):
    FM=np.zeros((n_resumes,n_features))
    self.FM=FM
    
  def feature_gen(self,matches,doclist,features):
    best_ai_score,best_wb_score,best_app_score=21,18,15  # best available resume taken for reference point for each domain
    target_data=[]
    for (i,match),doc in zip(enumerate(matches),doclist):
      feat_c=features.copy()  #copy of features
      print(f'for resume {i} \n')
      for match_id,pos1,pos2 in match: # since doc will never contain an unknown buzz word as spacy matcher has limited matches,therefore else condn is useless here
          # print(f'{nlp.vocab.strings[match_id]} : {doc[pos1:pos2].text}')

          if doc[pos1:pos2].text.lower() == 'machine learning' or doc[pos1:pos2].text.lower() == 'ml':
            feat_c['ML']+=1

          
          elif doc[pos1:pos2].text.lower() == 'artificial intelligence' or doc[pos1:pos2].text.lower() == 'ai':
            feat_c['AI']+=1

          elif doc[pos1:pos2].text.lower() == 'nlp' or doc[pos1:pos2].text.lower() == 'natural language processing':
            feat_c['NLP']+=1

          elif doc[pos1:pos2].text.lower() == 'deep learning':
            feat_c['DL']+=1

          elif doc[pos1:pos2].text.lower() == 'big data':
            feat_c['BD']+=1

          elif doc[pos1:pos2].text.lower() == 'reinforcement learning':
            feat_c['RL']+=1

          elif doc[pos1:pos2].text.lower() == 'cv' or doc[pos1:pos2].text.lower() == 'computer vision':
            feat_c['CV']+=1
          
          elif doc[pos1:pos2].text.lower() == 'data science':
            feat_c['data_science']+=1
          
          elif doc[pos1:pos2].text.lower() == 'data engineer':
            feat_c['data_engineer']+=1

          elif doc[pos1:pos2].text.lower() == 'data analyst':
            feat_c['data_analyst']+=1
          
          elif doc[pos1:pos2].text.lower() == 'unsupervised learning':
            feat_c['unsupervised_ML']+=1
          
          elif doc[pos1:pos2].text.lower() == 'supervised learning':
            feat_c['supervised_ML']+=1
          
          elif doc[pos1:pos2].text.lower() == 'neural networks':
            feat_c['NN']+=1
          
          elif doc[pos1:pos2].text.lower() == 'data mining':
            feat_c['data_mining']+=1
          
          elif doc[pos1:pos2].text.lower() == 'regression':
            feat_c['regression']+=1

          elif doc[pos1:pos2].text.lower() == 'classification':
            feat_c['classification']+=1

          elif doc[pos1:pos2].text.lower() == 'html':
            feat_c['HTML']+=1

          elif doc[pos1:pos2].text.lower() == 'css':
            feat_c['CSS']+=1

          elif doc[pos1:pos2].text.lower() == 'react':
            feat_c['react']+=1

          elif doc[pos1:pos2].text.lower() == 'javascript':
            feat_c['javascript']+=1

          elif doc[pos1:pos2].text.lower() == 'frontend':
            feat_c['frontend']+=1

          elif doc[pos1:pos2].text.lower() == 'backend':
            feat_c['backend']+=1
            
          elif doc[pos1:pos2].text.lower() == 'nodejs':
            feat_c['nodejs']+=1

          elif doc[pos1:pos2].text.lower() == 'firebase':
            feat_c['firebase']+=1

          elif doc[pos1:pos2].text.lower() == 'graphql':
            feat_c['GraphQL']+=1

          elif doc[pos1:pos2].text.lower() == 'seo':
            feat_c['SEO']+=1

          elif doc[pos1:pos2].text.lower() == 'kotlin':
            feat_c['kotlin']+=1   

          elif doc[pos1:pos2].text.lower() == 'react native':
            feat_c['react-native']+=1

          elif doc[pos1:pos2].text.lower() == 'android':
            feat_c['android']+=1

          elif doc[pos1:pos2].text.lower() == 'android studio':
            feat_c['android studio']+=1

          elif doc[pos1:pos2].text.lower() == 'ios':
            feat_c['ios']+=1

          elif doc[pos1:pos2].text.lower() == 'android sdk':
            feat_c['android-sdk']+=1

          elif doc[pos1:pos2].text.lower() == 'dart':
            feat_c['dart']+=1

          elif doc[pos1:pos2].text.lower() == 'android app':
            feat_c['android-app']+=1

          elif doc[pos1:pos2].text.lower() == 'flutter':
            feat_c['flutter']+=1

          else:
            pass


      for j in range(len(feat_c)):     # This loop contains all the features in order to prepare the dataset 
          if j==0:
            self.FM[i,j]=feat_c['ML']
          elif j==1:
            self.FM[i,j]=feat_c['DL']
          elif j==2:
            self.FM[i,j]=feat_c['NLP']
          elif j==3:
            self.FM[i,j]=feat_c['BD']
          elif j==4:
            self.FM[i,j]=feat_c['RL']
          elif j==5:
            self.FM[i,j]=feat_c['CV']
          elif j==6:
            self.FM[i,j]=feat_c['data_science']
          elif j==7:
            self.FM[i,j]=feat_c['data_engineer']
          elif j==8:
            self.FM[i,j]=feat_c['data_analyst']
          elif j==9:
            self.FM[i,j]=feat_c['AI']
          elif j==10:
            self.FM[i,j]=feat_c['unsupervised_ML']
          elif j==11:
            self.FM[i,j]=feat_c['supervised_ML']
          elif j==12:
            self.FM[i,j]=feat_c['NN']
          elif j==13:
            self.FM[i,j]=feat_c['data_mining']
          elif j==14:
            self.FM[i,j]=feat_c['regression']
          elif j==15:
            self.FM[i,j]=feat_c['classification']    
          elif j==16:
            self.FM[i,j]=feat_c['bayesian']      
          elif j==17:
            self.FM[i,j]=feat_c['HTML']
          elif j==18:
            self.FM[i,j]=feat_c['CSS']
          elif j==19:
            self.FM[i,j]=feat_c['javascript']
          elif j==20:
            self.FM[i,j]=feat_c['backend']
          elif j==21:
            self.FM[i,j]=feat_c['firebase']
          elif j==22:
            self.FM[i,j]=feat_c['frontend']
          elif j==23:
            self.FM[i,j]=feat_c['react']
          elif j==24:
            self.FM[i,j]=feat_c['nodejs']
          elif j==25:
            self.FM[i,j]=feat_c['GraphQL']
          elif j==26:
            self.FM[i,j]=feat_c['SEO']
          elif j==27:
            self.FM[i,j]=feat_c['kotlin']
          elif j==28:
            self.FM[i,j]==feat_c['android']
          elif j==29:
            self.FM[i,j]=feat_c['ios']
          elif j==30:
            self.FM[i,j]=feat_c['android studio']
          elif j==31:
            self.FM[i,j]=feat_c['react-native']
          elif j==32:
            self.FM[i,j]=feat_c['dart']
          elif j==33:
            self.FM[i,j]=feat_c['android-app']
          elif j==34:
            self.FM[i,j]=feat_c['android-sdk']
          elif j==35:
            self.FM[i,j]=feat_c['flutter']
          else:
            pass
          
      data=self.class_label(feat_c,best_ai_score,best_wb_score,best_app_score)
      target_data.append(data)

    self.target_data=target_data
    return self.FM,self.target_data

  def class_label(self,feat_c,best_ai_score,best_wb_score,best_app_score):
      #for AI
      y_data=[]
      ai_score=(feat_c['ML']+feat_c['DL']+feat_c['NLP']+feat_c['BD']+feat_c['RL']+feat_c['CV']+feat_c['AI']+feat_c['data_engineer']+feat_c['data_science']+
                feat_c['data_analyst']+feat_c['unsupervised_ML']+feat_c['supervised_ML']+feat_c['NN']+feat_c['data_mining']+feat_c['regression']+
                feat_c['classification']+feat_c['bayesian'])/best_ai_score       

      #for web unjuk

      wb_score=(feat_c['frontend']+feat_c['backend']+feat_c['react']+feat_c['javascript']+feat_c['HTML']+feat_c['CSS']+feat_c['nodejs']+
                feat_c['GraphQL']+feat_c['SEO']+feat_c['firebase'])/best_wb_score  

      #for app dev

      app_score=(feat_c['kotlin']+feat_c['android']+feat_c['ios']+feat_c['react-native']+feat_c['android studio']+feat_c['android-sdk']+
                 feat_c['android-app']+feat_c['dart']+feat_c['flutter'])/best_app_score
      
      if ai_score>wb_score and ai_score>app_score:
        y_data.append(f'AI developer : {ai_score}')
      elif wb_score>ai_score and wb_score>app_score:
        y_data.append(f'web developer : {wb_score}')
      elif app_score>ai_score and app_score>wb_score:
        y_data.append(f'app developer : {app_score}')                                     #Need to work on this logic

      
      return y_data


def patterns():

  nlp=spacy.load('en_core_web_sm')
  matcher=Matcher(nlp.vocab)

  ## buzz words

  #for ai 
  pattern_1= [{"LOWER": "machine"}, {"LOWER": "learning"}]
  pattern_2= [{"LOWER": "deep"}, {"LOWER": "learning"}]
  pattern_3= [{"LOWER": "nlp"}]
  pattern_4= [{"LOWER": "big"}, {"LOWER": "data"}]
  pattern_5= [{"LOWER": "reinforcement"}, {"LOWER": "learning"}]
  pattern_6= [{"LOWER": "supervised"}, {"LOWER": "learning"}]
  pattern_7= [{"LOWER": "data"}, {"LOWER": "science"}]
  pattern_8= [{"LOWER": "data"}, {"LOWER": "engineer"}]
  pattern_9= [{"LOWER": "data"}, {"LOWER": "analyst"}]
  pattern_10= [{"LOWER": "ai"}]
  pattern_11= [{"LOWER": "computer"}, {"LOWER": "vision"}]
  pattern_12= [{"LOWER": "ml"}]
  pattern_13= [{"LOWER": "cv"}]
  pattern_14= [{"LOWER": "natural"}, {"LOWER": "language"}, {"LOWER": "processing"}]
  pattern_15= [{"LOWER": "unsupervised"}, {"LOWER": "learning"}]
  pattern_16= [{"LOWER": "neural"}, {"LOWER": "networks"}]
  pattern_17= [{"LOWER": "data"}, {"LOWER": "mining"}]
  pattern_18= [{"LOWER": "regression"}]
  pattern_19= [{"LOWER": "classification"}]
  pattern_20= [{"LOWER": "bayesian"}]


  #for app dev
  pattern_21= [{"LOWER": "android"}, {"LOWER": "studio"}]
  pattern_22= [{"LOWER": "kotlin"}]
  pattern_23= [{"LOWER": "react"}, {"LOWER":"native"}]
  pattern_24= [{"LOWER": "ios"}]
  pattern_25= [{"LOWER": "android"}]        
  pattern_26= [{"LOWER": "android"}, {"LOWER": "app"}]
  pattern_27= [{"LOWER": "dart"}]
  pattern_28= [{"LOWER": "android"}, {"LOWER": "sdk"}]
  pattern_39= [{"LOWER": "flutter"}]

  #for web dev

  pattern_29= [{"LOWER": "frontend"}]
  pattern_30= [{"LOWER": "backend"}]
  pattern_31= [{"LOWER": "react"}]
  pattern_32= [{"LOWER": "nodejs"}]
  pattern_33= [{"LOWER": "javascript"}]
  pattern_34= [{"LOWER": "html"}]
  pattern_35= [{"LOWER": "css"}]
  pattern_36= [{"LOWER": "firebase"}]
  pattern_37= [{"LOWER": "graphql"}]
  pattern_38= [{"LOWER": "seo"}]


  matcher.add("AI_TEST_PATTERNS_1",[pattern_1])
  matcher.add("AI_TEST_PATTERNS_2",[pattern_2])
  matcher.add("AI_TEST_PATTERNS_3",[pattern_3])
  matcher.add("AI_TEST_PATTERNS_4",[pattern_4])
  matcher.add("AI_TEST_PATTERNS_5",[pattern_5])
  matcher.add("AI_TEST_PATTERNS_6",[pattern_6])
  matcher.add("AI_TEST_PATTERNS_7",[pattern_7])
  matcher.add("AI_TEST_PATTERNS_8",[pattern_8])
  matcher.add("AI_TEST_PATTERNS_9",[pattern_9])
  matcher.add("AI_TEST_PATTERNS_10",[pattern_10])
  matcher.add("AI_TEST_PATTERNS_11",[pattern_11])
  matcher.add("AI_TEST_PATTERNS_12",[pattern_12])
  matcher.add("AI_TEST_PATTERNS_13",[pattern_13])
  matcher.add("AI_TEST_PATTERNS_14",[pattern_14])
  matcher.add("AI_TEST_PATTERNS_15",[pattern_15])
  matcher.add("AI_TEST_PATTERNS_16",[pattern_16])
  matcher.add("AI_TEST_PATTERNS_17",[pattern_17])
  matcher.add("AI_TEST_PATTERNS_18",[pattern_18])
  matcher.add("AI_TEST_PATTERNS_19",[pattern_19])
  matcher.add("AI_TEST_PATTERNS_20",[pattern_20])
  matcher.add("App_TEST_PATTERNS_21",[pattern_21])
  matcher.add("App_TEST_PATTERNS_22",[pattern_22])
  matcher.add("App_TEST_PATTERNS_23",[pattern_23])
  matcher.add("App_TEST_PATTERNS_24",[pattern_24])
  matcher.add("App_TEST_PATTERNS_25",[pattern_25])
  matcher.add("App_TEST_PATTERNS_26",[pattern_26])
  matcher.add("App_TEST_PATTERNS_27",[pattern_27])
  matcher.add("App_TEST_PATTERNS_28",[pattern_28])
  matcher.add("App_TEST_PATTERNS_39",[pattern_39])
  matcher.add("wb_TEST_PATTERNS_29",[pattern_29])
  matcher.add("wb_TEST_PATTERNS_30",[pattern_30])
  matcher.add("wb_TEST_PATTERNS_31",[pattern_31])
  matcher.add("wb_TEST_PATTERNS_32",[pattern_32])
  matcher.add("wb_TEST_PATTERNS_33",[pattern_33])
  matcher.add("wb_TEST_PATTERNS_34",[pattern_34])
  matcher.add("wb_TEST_PATTERNS_35",[pattern_35])
  matcher.add("wb_TEST_PATTERNS_36",[pattern_36])
  matcher.add("wb_TEST_PATTERNS_37",[pattern_37])
  matcher.add("wb_TEST_PATTERNS_38",[pattern_38])

  features={'ML':0,'DL':0,'NLP':0,'BD':0,'RL':0,'CV':0,'AI':0,'data_science':0,'data_engineer':0,'data_analyst':0,
            'unsupervised_ML':0,'supervised_ML':0,'NN':0,'data_mining':0,'regression':0,'classification':0,'bayesian':0,
            'frontend':0,'backend':0,'react':0,'javascript':0,'HTML':0,'CSS':0,'nodejs':0,'firebase':0,'GraphQL':0,'SEO':0,
            'kotlin':0,'android':0,'ios':0,'android-sdk':0,'react-native':0,'android studio':0,'dart':0,'android-app':0,'flutter':0}

  return features,matcher


# if __name__=='__main__':

#   features,matcher=patterns()
#   resumelist=['Resume data/My resume optional.pdf','Resume data/My Resume.pdf','Resume data/rds.pdf']  #examples
#   resume_obj=Resume_Extractor(resumelist)
#   matches,doclist=resume_obj(matcher)
#   # print(Resume_Extractor.doc_list)
#   # for match_id,pos1,pos2 in matches: 
#   #   print(f'{nlp.vocab.strings[match_id]} : {doc[pos1:pos2].text}')


#   arr_obj=Feature_Matrix(len(resumelist),len(features))
#   x_data,y_data=arr_obj.feature_gen(matches,doclist,features)
  
#   print(len(features))
#   print(x_data)
#   print('\n')
#   print(y_data)

