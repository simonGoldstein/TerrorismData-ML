import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import helpers.modelSaving as modelSave
import numpy as np
import pandas as pd

knnModel = modelSave.loadSk('trainedModels/knn')

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Terrorism Perpetrator Predictor'),
    html.H2('Please enter the following details about the incident.'),
    html.Div([
        html.H3('Timing Information:'),
        html.Div([
            html.H5('Year'),
            dcc.Input(id='iyear', type='number')
        ]),
        html.Div([
            html.H5('Month'),
            dcc.Input(id='imonth', type='number', min=1, max=12)
        ]),
        html.Div([
            html.H5('Day'),
            dcc.Input(id='iday', type='number', min=1, max=31)
        ]),
        html.Div([
            html.H5('How long did the incident last?'),
            dcc.RadioItems(
                id='extended',
                options = [
                    {'label': 'Over 24 Hours', 'value': '1'},
                    {'label': '24 Hours or Less', 'value': '0'}
                ]
            )
        ])
    ]),
    html.Div([
        html.H3('Location Information:'),
        html.Div([
            html.H5('Country'),
            dcc.Input(id='country')
        ]),
        html.Div([
            html.H5('Region'),
            dcc.Input(id='region')
        ]),
        html.Div([
            html.H5('Did the incident occur within a city?'),
            dcc.RadioItems(
                id='vicinity',
                options = [
                    {'label': 'No: The indicent occured in the immediate vicinity of the city.', 'value': '1'},
                    {'label': 'Yes: The incident occured in the city itself.', 'value': '0'}
                ]
            )
        ])
    ]),
    html.Div([
        html.H3('Attack Information:'),
        html.Div([
            html.H5('Was the attack successful? (Did it actualy take place?)'),
            dcc.RadioItems(
                id='success',
                options = [
                    {'label': 'No', 'value': '0'},
                    {'label': 'Yes', 'value': '1'}
                ]
            )
        ]),
        html.Div([
            html.H5('Did the attacker commit suicide?'),
            dcc.RadioItems(
                id='suicide',
                options = [
                    {'label': 'No', 'value': '0'},
                    {'label': 'Yes', 'value': '1'}
                ]
            )
        ]),
        html.Div([
            html.H5('Attack Type'),
            dcc.Input(id='attacktype1')
        ]),
        html.Div([
            html.H5('Target Type'),
            dcc.Input(id='targtype1')
        ]),
        html.Div([
            html.H5('Weapon Type'),
            dcc.Input(id='weaptype1')
        ]),
        html.Div([
            html.H5('Did the attack result in property damage?'),
            dcc.RadioItems(
                id='property',
                options = [
                    {'label': 'Yes', 'value': '1'},
                    {'label': 'No', 'value': '0'},
                    {'label': 'Unknown', 'value': '-9'}
                ]
            )
        ]),
        html.Div([
            html.H5('Were the attacker(s) identified as individuals (unaffiliated with a terrorist group)?'),
            dcc.RadioItems(
                id='individual',
                options = [
                    {'label': 'Yes: The attackers definitely were not associated with a group.', 'value': '1'},
                    {'label': 'No: The attackers were either definitely working with a group, or it is unknown whether or not they were working with a group.', 'value': '0'},
                ]
            )
        ]),
        html.Div([
            html.H5('Was the attack Logistically International?'),
            dcc.RadioItems(
                id='INT_LOG',
                options = [
                    {'label': 'Yes: The attack was logistically international; the nationality of the \
perpetrator group differs from the location of the attack. If \
the perpetrator group is multinational, the attack is logistically \
international if all of the group’s nationalities differ from the \
location of the attack.', 'value': '1'},
                    {'label': 'No: The attack was logistically domestic; the nationality of the \
perpetrator group is the same as the location of the attack. If \
the perpetrator group is multinational, the attack is logistically \
domestic if any of the group’s nationalities is the same as the \
location of the attack.', 'value': '0'},
                    {'label': 'Unknown', 'value': '-9'}
                ]
            )
        ]),
        html.Div([
            html.H5('Was the attack Ideologically International?'),
            dcc.RadioItems(
                id='INT_IDEO',
                options = [
                    {'label': 'Yes: The attack was ideologically international; the nationality of \
the perpetrator group differs from the nationality of the \
target(s)/victim(s). If the perpetrator group or target is \
multinational, the attack is ideologically international.', 'value': '1'},
                    {'label': 'No: The attack was ideologically domestic; any and all nationalities \
of the perpetrator group are the same as the nationalities of \
the target(s)/victim(s). ', 'value': '0'},
                    {'label': 'Unknown', 'value': '-9'}
                ]
            )
        ]),
        html.Div([
            html.H5('Was the attack International (Misc)?'),
            html.P(' If an attack is international on this dimension, it \
is necessarily also either logistically international or ideologically international, but it \
is not clear which one.'),
            dcc.RadioItems(
                id='INT_MISC',
                options = [
                    {'label': 'Yes', 'value': '1'},
                    {'label': 'No', 'value': '0'},
                    {'label': 'Unknown', 'value': '-9'}
                ]
            )
        ]),

        html.Button('Submit', id='submit', style={'font-size': 'medium'}),

        html.Div([
            html.H4('The predicted terrorist group is:'),
            html.H3(id='gname')
        ], id='results', hidden=True)
    ])
])

@app.callback([Output('results', 'hidden'),
               Output('gname', 'children')],
              [Input('submit', 'n_clicks')],
              [State('iyear', 'value'),
               State('imonth', 'value'),
               State('iday', 'value'),
               State('extended', 'value'),
               State('country', 'value'),
               State('region', 'value'),
               State('vicinity', 'value'),
               State('success', 'value'),
               State('suicide', 'value'),
               State('attacktype1', 'value'),
               State('targtype1', 'value'),
               State('weaptype1', 'value'),
               State('property', 'value'),
               State('individual', 'value'),
               State('INT_LOG', 'value'),
               State('INT_IDEO', 'value'),
               State('INT_MISC', 'value')])
def updateResults(n_clicks, iyear, imonth, iday, 
                  extended, country, region, 
                  vicinity, success, suicide, 
                  attacktype1, targtype1, weaptype1, 
                  property, individual, INT_LOG, 
                  INT_IDEO, INT_MISC):

    print("Year:", iyear)
    print("Month:", imonth)
    print("Day:", iday)
    print("Extended:", extended)
    print("Country:", country)
    print("Region:", region)
    print("Vicinity:", vicinity)
    print("Success:", success)
    print("Suicide:", suicide)
    print("Attack Type:", attacktype1)
    print("Target Type:", targtype1)
    print("Weapon Type:", weaptype1)
    print("Property Damage:", property)
    print("Individual:", individual)
    print("Logistically International:", INT_LOG)
    print("Ideologically International:", INT_IDEO)
    print("International (Misc):", INT_MISC)

    INT_ANY = -1
    if INT_LOG == 1 or INT_IDEO == 1 or INT_MISC == 1:
        INT_ANY = 1
    elif INT_LOG == 0 and INT_IDEO == 0 and INT_MISC == 0:
        INT_ANY = 0
    else:
        INT_ANY = -9

    array = np.array([iyear, imonth, iday, extended, country,
                      region, vicinity, success, suicide, attacktype1,
                      targtype1, individual, weaptype1, property, 
                      INT_LOG, INT_IDEO, INT_MISC, INT_ANY])
    array = array.reshape(1, -1)

    if np.any(array == None):
        raise dash.exceptions.PreventUpdate
    else:
        prediction = knnModel.predict(array)
        return False, prediction

if __name__ == '__main__':
    app.run_server(debug=True)