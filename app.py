import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import helpers.modelSaving as modelSave
import numpy as np
import pandas as pd

knnModel = modelSave.loadSk('trainedModels/knn')
svmModel = modelSave.loadSk('trainedModels/svm-20000')

external_stylesheets = ["https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1('Terrorism Perpetrator Predictor'),
    html.H2('Please enter the following details about the incident.'),

    html.Div([
        html.Div([
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
                            {'label': 'Over 24 Hours', 'value': 1},
                            {'label': '24 Hours or Less', 'value': 0}
                        ]
                    )
                ])
            ], className='card-content')
        ], className='card'),
        html.Div([
            html.Div([
                html.H3('Location Information:'),
                html.Div([
                    html.H5('Country'),
                    dcc.Dropdown(
                        options=eval(open('assets/countries.txt').read()),
                        id='country'
                    )
                ]),
                html.Div([
                    html.H5('Region'),
                    html.Button([
                        html.I(
                            '\tHelp',
                            className='fa fa-info'
                        )
                    ], className='btn', id='regions-help-btn'),
                    dbc.Modal([
                        dbc.ModalHeader('Regions Info'),
                        dbc.ModalBody([
                            html.H4("North America"),
                            html.H5("Canada, Mexico, United States"),
                            html.Br(),
                            html.H4('Central America & Caribbean'),
                            html.H5('Antigua and Barbuda, Bahamas, Barbados, Belize, Cayman Islands, Costa Rica, Cuba, Dominica, Dominican Republic, El Salvador, Grenada, Guadeloupe, Guatemala, Haiti,  Honduras, Jamaica, Martinique, Nicaragua, Panama, St. Kitts and Nevis, St. Lucia, Trinidad and Tobago'),
                            html.Br(),
                            html.H4('South America'),
                            html.H5('Argentina, Bolivia, Brazil, Chile, Colombia, Ecuador, Falkland Islands, French Guiana, Guyana, Paraguay, Peru, Suriname, Uruguay, Venezuela'),
                            html.Br(),
                            html.H4('East Asia'),
                            html.H5('China, Hong Kong, Japan, Macau, North Korea, South Korea, Taiwan'),
                            html.Br(),
                            html.H4('Southeast Asia'),
                            html.H5('Brunei, Cambodia, East Timor, Indonesia, Laos, Malaysia, Myanmar, Philippines, Singapore, South Vietnam, Thailand, Vietnam'),
                            html.Br(),
                            html.H4('South Asia'),
                            html.H5('Afghanistan, Bangladesh, Bhutan, India, Maldives, Mauritius, Nepal, Pakistan, Sri Lanka'),
                            html.Br(),
                            html.H4('Central Asia'),
                            html.H5('Armenia, Azerbaijan, Georgia, Kazakhstan, Kyrgyzstan, Tajikistan, Turkmenistan, Uzbekistan'),
                            html.Br(),
                            html.H4('Western Europe'),
                            html.H5('Andorra, Austria, Belgium, Cyprus, Denmark, Finland, France, Germany, Gibraltar, Greece, Iceland, Ireland, Italy, Luxembourg, Malta, Netherlands, Norway, Portugal, Spain, Sweden, Switzerland, United Kingdom, Vatican City, West Germany (FRG)'),
                            html.Br(),
                            html.H4('Eastern Europe'),
                            html.H5('Albania, Belarus, Bosnia-Herzegovina, Bulgaria, Croatia, Czech Republic, Czechoslovakia, East Germany (GDR), Estonia, Hungary, Kosovo, Latvia, Lithuania, Macedonia, Moldova, Montenegro, Poland, Romania, Russia, Serbia, SerbiaMontenegro, Slovak Republic, Slovenia, Soviet Union, Ukraine, Yugoslavia'),
                            html.Br(),
                            html.H4('Middle East & North Africa'),
                            html.H5('Algeria, Bahrain, Egypt, Iran, Iraq, Israel, Jordan, Kuwait, Lebanon, Libya, Morocco, North Yemen, Qatar, Saudi Arabia, South Yemen, Syria, Tunisia, Turkey, United Arab Emirates, West Bank and Gaza Strip, Western Sahara, Yemen'),
                            html.Br(),
                            html.H4('Sub-Saharan Africa'),
                            html.H5('Angola, Benin, Botswana, Burkina Faso, Burundi, Cameroon, Central African Republic, Chad, Comoros, Democratic Republic of the Congo, Djibouti, Equatorial Guinea, Eritrea, Ethiopia, Gabon, Gambia, Ghana, Guinea, Guinea-Bissau, Ivory Coast, Kenya, Lesotho, Liberia, Madagascar, Malawi, Mali, Mauritania, Mozambique, Namibia, Niger, Nigeria, People\'s Republic of the Congo, Republic of the Congo, Rhodesia, Rwanda, Senegal, Seychelles, Sierra Leone, Somalia, South Africa, South Sudan, Sudan, Swaziland, Tanzania, Togo, Uganda, Zaire, Zambia, Zimbabwe'),
                            html.Br(),
                            html.H4('Australasia & Oceania'),
                            html.H5('Australia, Fiji, French Polynesia, New Caledonia, New Hebrides, New Zealand, Papua New Guinea, Solomon Islands, Vanuatu, Wallis and Futuna'),
                            html.Br()
                        ]),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="close-regions-help", className="ml-auto")
                        )],
                        id='regions-help',
                        centered=True,
                        scrollable=True,
                    ),
                    dcc.Dropdown(
                        options=eval(open('assets/regions.txt').read()),
                        id='region'
                    )
                ]),
                html.Div([
                    html.H5('Did the incident occur within a city?'),
                    dcc.RadioItems(
                        id='vicinity',
                        options = [
                            {'label': 'No: The indicent occured in the immediate vicinity of the city.', 'value': 1},
                            {'label': 'Yes: The incident occured in the city itself.', 'value': 0}
                        ]
                    )
                ])
            ], className='card-content')
        ], className='card'),
        html.Div([
            html.Div([
                html.H3('Attack Information:'),
                html.Div([
                    html.H5('Was the attack successful? (Did it actualy take place?)'),
                    dcc.RadioItems(
                        id='success',
                        options = [
                            {'label': 'No', 'value': 0},
                            {'label': 'Yes', 'value': 1}
                        ]
                    )
                ]),
                html.Div([
                    html.H5('Did the attacker commit suicide?'),
                    dcc.RadioItems(
                        id='suicide',
                        options = [
                            {'label': 'No', 'value': 0},
                            {'label': 'Yes', 'value': 1}
                        ]
                    )
                ]),
                html.Div([
                    html.H5('Attack Type'),
                    html.Button([
                        html.I(
                            '\tHelp',
                            className='fa fa-info'
                        )
                    ], className='btn', id='attack-type-help-btn'),
                    dbc.Modal([
                        dbc.ModalHeader('Attack Type Info'),
                        dbc.ModalBody([
                            html.H4("Assassination"),
                            html.H5("An act whose primary objective is to kill one or more specific, prominent individuals. Usually carried out on persons of some note, such as highranking military officers, government officials, celebrities, etc. Not to include attacks on non-specific members of a targeted group. The killing of a police officer would be an armed assault unless there is reason to believe the attackers singled out a particularly prominent officer for assassination."),
                            html.Br(),
                            html.H4('Armed Assault'),
                            html.H5("An attack whose primary objective is to cause physical harm or death directly to human beings by use of a firearm, incendiary, or sharp instrument (knife, etc.). Not to include attacks involving the use of fists, rocks, sticks, or other handheld (less-than-lethal) weapons. Also includes attacks involving certain classes of explosive devices in addition to firearms, incendiaries, or sharp instruments. The explosive device subcategories that are included in this classification are grenades, projectiles, and unknown or other explosive devices that are thrown."),
                            html.Br(),
                            html.H4('Bombing/Explosion'),
                            html.H5("An attack where the primary effects are caused by an energetically unstable material undergoing rapid decomposition and releasing a pressure wave that causes physical damage to the surrounding environment. Can include either high or low explosives (including a dirty bomb) but does not include a nuclear explosive device that releases energy from fission and/or fusion, or an incendiary device where decomposition takes place at a much slower rate. If an attack involves certain classes of explosive devices along with firearms, incendiaries, or sharp objects, then the attack is coded as an armed assault only. The explosive device subcategories that are included in this classification are grenades, projectiles, and unknown or other explosive devices that are thrown in which the bombers are also using firearms or incendiary devices."),
                            html.Br(),
                            html.H4('Hijacking'),
                            html.H5("An act whose primary objective is to take control of a vehicle such as an aircraft, boat, bus, etc. for the purpose of diverting it to an unprogrammed destination, force the release of prisoners, or some other political objective. Obtaining payment of a ransom should not the sole purpose of a Hijacking, but can be one element of the incident so long as additional objectives have also been stated. Hijackings are distinct from Hostage Taking because the target is a vehicle, regardless of whether there are people/passengers in the vehicle."),
                            html.Br(),
                            html.H4('Hostage Taking (Barricade Incident)'),
                            html.H5("An act whose primary objective is to take control of hostages for the purpose of achieving a political objective through concessions or through disruption of normal operations. Such attacks are distinguished from kidnapping since the incident occurs and usually plays out at the target location with little or no intention to hold the hostages for an extended period in a separate clandestine location. "),
                            html.Br(),
                            html.H4('Facility/Infrastructure Attack'),
                            html.H5("An act, excluding the use of an explosive, whose primary objective is to cause damage to a non-human target, such as a building, monument, train, pipeline, etc. Such attacks include arson and various forms of sabotage (e.g., sabotaging a train track is a facility/infrastructure attack, even if passengers are killed). Facility/infrastructure attacks can include acts which aim to harm an installation, yet also cause harm to people incidentally (e.g. an arson attack primarily aimed at damaging a building, but causes injuries or fatalities)."),
                            html.Br(),
                            html.H4('Unarmed Assault'),
                            html.H5("An attack whose primary objective is to cause physical harm or death directly to human beings by any means other than explosive, firearm, incendiary, or sharp instrument (knife, etc.). Attacks involving chemical, biological or radiological weapons are considered unarmed assaults."),
                            html.Br(),
                            html.H4('Unknown'),
                            html.H5("The attack type cannot be determined from the available information."),
                            html.Br()
                        ]),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="close-attack-type-help", className="ml-auto")
                        )],
                        id='attack-type-help',
                        centered=True,
                        scrollable=True,
                    ),
                    dcc.Dropdown(
                        options=[
                            {'label': 'Assassination', 'value': 1},
                            {'label': 'Armed Assault', 'value': 2},
                            {'label': 'Bombing / Explosion', 'value': 3},
                            {'label': 'Hijacking', 'value': 4},
                            {'label': 'Hostage Taking (Barricade Incident)', 'value': 5},
                            {'label': 'Hostage Taking (Kidnapping)', 'value': 6},
                            {'label': 'Facility / Infrastructure Attack', 'value': 7},
                            {'label': 'Unarmed Assault', 'value': 8},
                            {'label': 'Unknown', 'value': 9}
                        ],
                        id='attacktype1'
                    )
                ]),

                html.Div([
                    html.H5('Target Type'),
                    html.Button([
                        html.I(
                            '\tHelp',
                            className='fa fa-info'
                        )
                    ], className='btn', id='target-help-btn'),
                    dbc.Modal([
                        dbc.ModalHeader('Targets Info'),
                        dbc.ModalBody(eval(open('assets/targetHelp.txt').read())),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="close-target-help", className="ml-auto")
                        )],
                        id='target-help',
                        centered=True,
                        scrollable=True,
                    ),
                    dcc.Dropdown(
                        options=eval(open('assets/targets.txt').read()),
                        id='targtype1'
                    )
                ]),
                html.Div([
                    html.H5('Weapon Type'),
                    html.Button([
                        html.I(
                            '\tHelp',
                            className='fa fa-info'
                        )
                    ], className='btn', id='weapons-help-btn'),
                    dbc.Modal([
                        dbc.ModalHeader('Weapons Info'),
                        dbc.ModalBody(eval(open('assets/weaponsHelp.txt').read())),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="close-weapons-help", className="ml-auto")
                        )],
                        id='weapons-help',
                        centered=True,
                        scrollable=True,
                    ),
                    dcc.Dropdown(
                        options=eval(open('assets/weapons.txt').read()),
                        id='weaptype1'
                    )
                ]),

                html.Div([
                    html.H5('Did the attack result in property damage?'),
                    dcc.RadioItems(
                        id='property',
                        options = [
                            {'label': 'Yes', 'value': 1},
                            {'label': 'No', 'value': 0},
                            {'label': 'Unknown', 'value': -9}
                        ]
                    )
                ]),
                html.Div([
                    html.H5('Were the attacker(s) identified as individuals (unaffiliated with a terrorist group)?'),
                    dcc.RadioItems(
                        id='individual',
                        options = [
                            {'label': 'Yes: The attackers definitely were not associated with a group.', 'value': 1},
                            {'label': 'No: The attackers were either definitely working with a group, or it is unknown whether or not they were working with a group.', 'value': 0},
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
        location of the attack.', 'value': 1},
                            {'label': 'No: The attack was logistically domestic; the nationality of the \
        perpetrator group is the same as the location of the attack. If \
        the perpetrator group is multinational, the attack is logistically \
        domestic if any of the group’s nationalities is the same as the \
        location of the attack.', 'value': 0},
                            {'label': 'Unknown', 'value': -9}
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
        multinational, the attack is ideologically international.', 'value': 1},
                            {'label': 'No: The attack was ideologically domestic; any and all nationalities \
        of the perpetrator group are the same as the nationalities of \
        the target(s)/victim(s). ', 'value': 0},
                            {'label': 'Unknown', 'value': -9}
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
                            {'label': 'Yes', 'value': 1},
                            {'label': 'No', 'value': 0},
                            {'label': 'Unknown', 'value': -9}
                        ]
                    )
                ])
            ], className='card-content')
        ], className='card'),

        html.Div([
            html.Div([
                html.H3('Results'),

                html.H4('Select Model:'),
                dcc.RadioItems(
                    id='model',
                    options=[{'label': 'KNN', 'value': 'knn'},
                             {'label': 'SVM', 'value': 'svm'}]
                ),

                html.Button('Compute', id='submit', style={'font-size': 'medium'}),

                html.Div([
                    'There is a field missing!'
                ], id='error', hidden=True, style={'color': 'red'}),

                html.Div([
                    html.H4('The predicted terrorist group is:'),
                    html.H3(id='gname')
                ], id='results', hidden=True)
            ], className='card-content')
        ], className='card')
    ])
])

def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


app.callback(
    Output("regions-help", "is_open"),
    [Input("regions-help-btn", "n_clicks"),
    Input("close-regions-help", "n_clicks")],
    [State("regions-help", "is_open")],
)(toggle_modal)

app.callback(
    Output("attack-type-help", "is_open"),
    [Input("attack-type-help-btn", "n_clicks"),
    Input("close-attack-type-help", "n_clicks")],
    [State("attack-type-help", "is_open")],
)(toggle_modal)

app.callback(
    Output("target-help", "is_open"),
    [Input("target-help-btn", "n_clicks"),
    Input("close-target-help", "n_clicks")],
    [State("target-help", "is_open")],
)(toggle_modal)

app.callback(
    Output("weapons-help", "is_open"),
    [Input("weapons-help-btn", "n_clicks"),
    Input("close-weapons-help", "n_clicks")],
    [State("weapons-help", "is_open")],
)(toggle_modal)

@app.callback([Output('iyear', 'value'),
               Output('imonth', 'value'),
               Output('iday', 'value'),
               Output('extended', 'value'),
               Output('country', 'value'),
               Output('region', 'value'),
               Output('vicinity', 'value'),
               Output('success', 'value'),
               Output('suicide', 'value'),
               Output('attacktype1', 'value'),
               Output('targtype1', 'value'),
               Output('weaptype1', 'value'),
               Output('property', 'value'),
               Output('individual', 'value'),
               Output('INT_LOG', 'value'),
               Output('INT_IDEO', 'value'),
               Output('INT_MISC', 'value')],
              [Input('submit', 'n_clicks')])
def initialize(n_clicks):
    if not n_clicks:
        return 1992, 12, 1, 1, 14, 12, 0, 1, 1, 3, 2, 1, -9, 1, 1, 0, 0
    else:
        raise dash.exceptions.PreventUpdate

@app.callback([Output('results', 'hidden'),
               Output('gname', 'children'),
               Output('error', 'hidden')],
              [Input('submit', 'n_clicks')],
              [State('model', 'value'),
               State('iyear', 'value'),
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
def updateResults(n_clicks, model, iyear, imonth, iday, 
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
        return True, 0, False
    else:
        if model == 'knn':
            prediction = knnModel.predict(array)
            return False, prediction, True
        elif model == 'svm':
            prediction = svmModel.predict(array)
            return False, prediction, True

if __name__ == '__main__':
    app.run_server(debug=True)