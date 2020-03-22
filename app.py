#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 08:59:34 2020

@author: bala
"""
#libraries for the webapp
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State


#libraries for web scraping and misc
import warnings
warnings.filterwarnings('ignore')
import requests
from bs4 import BeautifulSoup

#libraries from pyTorch + pre-trained ALBERT model
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

#tokenizer = AutoTokenizer.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2")
#model = AutoModelForQuestionAnswering.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2")

tokenizer = AutoTokenizer.from_pretrained("/home/bala/Documents/Caliber2020/1/pytorch_model.bin")
model = AutoModelForQuestionAnswering.from_pretrained("/home/bala/Documents/Caliber2020/1/pytorch_model.bin")

      

###################################################
#layout section
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

pyTorchLogo = 'https://www.google.com/url?sa=i&url=https%3A%2F%2Fcommons.wikimedia.org%2Fwiki%2FFile%3APytorch_logo.png&psig=AOvVaw2ahWLA27Kt_ZCgvOiL8aJM&ust=1584769345965000&source=images&cd=vfe&ved=2ahUKEwjQ4c2ArKjoAhUGkksFHSKWDAQQr4kDegUIARD8AQ'

app.layout = html.Div([
    html.Div([
        html.Img(src=app.get_asset_url('Azure logo.png'),
                 style={'width':'20%'}),
        html.Img(src=app.get_asset_url('pythonPyTorchTensorFlow.jpeg'),
                 style={'width':'20%'}),
        dcc.Markdown('''## Caliber2020/ PreTrainedModels''')
            ], style={'textAlign':'center','columnCount':2, 
                    'backgroundColor':'#7a0404', 'color':'#ffffff',
                    #'background-image': 'url(https://upload.wikimedia.org/wikipedia/commons/2/22/North_Star_-_invitation_background.png'
                    }
            ),
    html.Div([
        html.Div([
            dcc.Markdown('''### Enter full URL for the passage you want to query'''),
            dcc.Input(
                id='url',
                value='https://docs.microsoft.com/en-us/azure/cloud-adoption-framework/',
                type='text',
                style={'width':'60%'}
            ),
            dcc.Markdown('''### Enter your question below'''),
            dcc.Input(
                id='question',
                value='Explain cloud adoption framework',
                type='text',
                style={'width':'60%'}
            ),
            html.Button(id='button', n_clicks=0, children='Submit',
                        style={'color':'#ffffff', 'backgroundColor':'#7a0404'}),
            
            
                ], style={'columnCount':1,
            #'background-image': 'url(https://upload.wikimedia.org/wikipedia/commons/2/22/North_Star_-_invitation_background.png'
                    }
                ),
        dcc.Markdown('''### This is the answer generated'''),
        dcc.Markdown('''*(Please wait for a moment to see answer below)*'''),
        html.Div(id='answer')
            ], style={'columnCount':1, 'backgroundColor':'#e2e3de'}
            ),
        
   
    dcc.Markdown('''---'''),
    dcc.Markdown('''*This Q & A service is powered by an ALBERT model pretrained on the
		SQuAD2.0 Q & A dataset, which has relatively short passages. If a longer piece
		of text (>512 words) is web scraped, this service errors out. 
		To minimize the chance of this, only the first 510 words are analyzed. 
		If this service cannot answer your question, please try a different URL or 			question.
		When this model is trained for our specific dataset (not SQuAD), model training 		configuration will need to be changed. For example, by using a sliding window
		as explained [here](https://github.com/google-research/bert/issues/66).*''')
    ])
########################################################
#callback section
@app.callback(
    Output(component_id='answer', component_property='children'),
    [Input(component_id='button', component_property='n_clicks')],
     [State(component_id='url', component_property='value'),
     State(component_id='question', component_property='value')
     ]
        )
def hfalbertqna (n_clicks, url, question):
    #reading from an URL
    res = requests.get(url)
    html_page = res.content
    soup = BeautifulSoup(html_page, 'html.parser')
    text = soup.find_all(text=True)
    
    passage5 = ''
    blacklist = [
    	'[document]',
    	'noscript',
    	'header',
    	'html',
    	'meta',
    	'head', 
    	'input',
    	'script',
    ]
    
    for t in text:
    	if t.parent.name not in blacklist:
    		passage5 += '{} '.format(t)
    passage = ' '.join(passage5.split())[:510]
    
    input_dict = tokenizer.encode_plus(question, passage, return_tensors="pt")
    input_ids = input_dict["input_ids"].tolist()
    start_scores, end_scores = model(**input_dict)
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    answer = ''.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]).replace('▁', ' ').strip()
    return(answer)


########################################################
#main
if __name__ == '__main__':
    app.run_server(host='0.0.0.0',debug=True, port=8050)

