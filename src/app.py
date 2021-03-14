import dash
import dash_html_components as html
import dash_core_components as dcc
import altair as alt
import pandas as pd
import dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from datetime import datetime, timedelta
import plotly as py
import plotly.express as px
import plotly.graph_objects as go
from flask_sqlalchemy import SQLAlchemy
from flask import Flask
from scipy import stats

alt.data_transformers.disable_max_rows()

########## Additional Data Filtering ###########################################
df = pd.read_csv('data/processed/business_data.csv', sep=';') #data/processed/cleaned_data.csv

###############################################################################



def create_card(header='Header', content='Card Content'): 
    
    if header:
        card = dbc.Card([dbc.CardHeader(header),
            dbc.CardBody(html.Label([content]))])
    elif not header:
        card = dbc.Card([dbc.CardBody(html.Label([content]))])
    return card 


def within_thresh(value, businesstype, column, data, sd_away=1):
    '''returns the lower and upper thresholds and whether the input
       falls within this threshold
    '''
    if column == 'Total Fees Paid':
        a = data.groupby('BusinessName').sum().reset_index()
        b = data.loc[:,['BusinessName','BusinessType']]
        data = pd.merge(a, b, how="left", on="BusinessName").drop_duplicates()
        column = 'FeePaid'

    mean_df = data.groupby('BusinessType').mean()
    sd_df = data.groupby('BusinessType').std()
    
    mean = mean_df.loc[businesstype, column]
    sd = float(sd_df.loc[businesstype, column])
    
    print(sd)
    upper_thresh = mean + sd_away*sd 
    lower_thresh = mean - sd_away*sd

    if lower_thresh < 0:
        lower_thresh = 0 

    if value > upper_thresh or value < lower_thresh: 
        return lower_thresh, upper_thresh, False
    else: 
        return lower_thresh, upper_thresh, True


server = Flask(__name__)
app = dash.Dash(__name__ , external_stylesheets=[dbc.themes.BOOTSTRAP], server=server)
app.title = "Fraudulent Business Detection"
app.server.config['SQLALCHEMY_DATABASE_URI'] = "postgres+psycopg2://postgres:one-percent@130.211.113.135:5432/postgres"
app.server.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app.server)


business_info = pd.read_sql_table('business_info', con=db.engine)
business_info = business_info.rename(columns={'business_name': 'BusinessName'})


app = dash.Dash(__name__ , external_stylesheets=[dbc.themes.BOOTSTRAP])
# Set the app title
app.title = "Fraudulent Buisness Detection"
server=app.server

colors = {
    'background': "#00000",
    'text': '#522889'
}

collapse = html.Div(
    [
        dbc.Button(
            "Learn more",
            id="collapse-button",
            className="mb-3",
            outline=False,
            style={'margin-top': '10px',
                'width': '150px',
                'background-color': 'white',
                'color': '#522889'}
        ),
    ]
)

@app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open



app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1('Fraudulent Buisness Detection', style={'text-align': 'center', 'color': 'white', 'font-size': '40px', 'font-family': 'Georgia'}),
            dbc.Collapse(html.P(
                """
                The dashboard will help you with your wine shopping today. Whether you desire crisp Californian Chardonnay or bold Cabernet Sauvignon from Texas, simply select a state and the wine type. The results will help you to choose the best wine for you.
                """,
                style={'color': 'white', 'width': '70%'}
            ), id='collapse'),
        ], md=10),
        dbc.Col([collapse])
    ], style={'backgroundColor': '#0F5DB6', 'border-radius': 3, 'padding': 15, 'margin-top': 22, 'margin-bottom': 22, 'margin-right': 11}),

    # dcc.Tabs([
        dcc.Tab([
            dbc.Row([
                dbc.Col([
                    html.Br(),
                    html.Label([
                        'Company Name'], style={
                'color': '#0F5DB6', "font-weight": "bold"
            }),
                    dcc.Dropdown(
                        id='business-name',
                        options=[{'label': name, 'value': name} for name in list(df['BusinessName'].dropna().unique())],
                        style={'width': '100%', 'height': 30},
                        placeholder='Select a State',
                        value = 'Time Education Inc'

                    ),
                    html.Br(),
                    html.Label(['Street Address'], style={'color': '#0F5DB6', "font-weight": "bold"}
                    ),
                    html.Br(),
                    html.Label(id='address'),  # Not capturing unit number
                    html.Br(),
                    html.Label(['Search Url'], style={'color': '#0F5DB6', "font-weight": "bold"}),
                    dcc.Textarea(style={'width': '100%', 'height': 30}),
                    dbc.Button('Web Search', id = 'scrape-btn', n_clicks=0, className='reset-btn-1'),
                    ], style={'border': '1px solid', 'border-radius': 3, 'padding': 15, 'margin-top': 22, 'margin-bottom': 15, 'margin-right': 0, 'height' : 350}, md=4,
                ),
                dbc.Col([], md=1),
                dbc.Col([
                    html.Br(),
                    html.Br(),
                    dbc.Row([dbc.Card([
                        dbc.CardHeader('Features Beyond 1-SD away from mean'),
                        dbc.CardBody(id='score', style={'color': '#0F5DB6', 'fontSize': 18,  'height': '70px'}),
                    ]
                    )]),
                    html.Br(),
                    dbc.Row([create_card(None,'This will be the number of outlier pts')]),
                ], md = 2),
                dbc.Col([
                    dcc.Graph(id='pie-chart',
                             figure = {'layout': go.Layout(margin={'b': 0})})
                ],)
                ]),
            dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader('Key Insights:', 
                            style={'fontWeight': 'bold', 'color':'white','font-size': '22px', 'backgroundColor':'#0F5DB6', 'height': '50px'}),
                            dbc.CardBody(id='insight-1', style={'color': '#0F5DB6', 'fontSize': 18,  'height': '70px'}),
                            html.Br(),
                            dbc.CardBody(id='insight-2', style={'color': '#0F5DB6', 'fontSize': 18,  'height': '380px'}),
                        ]),
                    ], md = 4),
                    dbc.Col([], md = 2),
                    dbc.Col([
                    dbc.Row([
                            dcc.Graph(id='histogram'),         
                        ]),
                        dcc.Dropdown(
                                id='feature_type',
                                value='Number of Employees',
                                options = [{'label': col, 'value': col} for col in ['Fee Paid', 'Number of Employees', 'Total Fees Paid', 'Missing Values']],
                                placeholder='Select a Feature', 
                                multi=False
                            ),
                        dcc.Dropdown(
                                id='std',
                                options = [{'label': col, 'value': col} for col in [1,2,3]],
                                placeholder='Select a standard dev', 
                                value='',
                                multi=False
                            ),
                    html.Br(), 
                    ],md = 6),
                ]),
            ]),
    ])


@app.callback(Output('insight-1', 'children'),
             Input('business-name', 'value'))
def url_presence(business):
    website = 'www.google.com'
    # business_df = df.query('BusinessName == @business')
    # website = business_df.iloc[-1, 'website']
    if website:
        insight = f"Website: {website}"
    if website:
        insight = 'No website available'
    return insight

@app.callback(Output('insight-2', 'children'),
             Input('business-name', 'value'))
def time_online(business):
    toy1 = {'BusinessName':"Time Education Inc", 'url' : 'www.google.ca'}
    toy2 = {'url' : 'www.google.ca', 'register_date' : '1999-01-02', 'expire_date': '2022-09-29'}

    url_df = pd.DataFrame(toy1, index=[1])
    url_info = pd.DataFrame(toy2, index=[1])

    filtered_url_df = url_df.query('BusinessName == @business')
    url = url_df.loc[1, 'url']
    url_time_df =url_info.query('url == @url')
    url_time_df = url_time_df.set_index('url')
    reg_time = pd.to_datetime(url_time_df.loc[url, 'register_date'])
    reg_time = reg_time.strftime('%B') + ' ' + reg_time.strftime('%Y')
    exp_time = pd.to_datetime(url_time_df.loc[url, 'expire_date'])
    if datetime.now() < exp_time:
        conj = 'has'
        exp_time = 'present'
    else: 
        conj = 'was'
        exp_time = pd.to_datetime(url_time_df.loc['url', 'expire_date'])
        exp_time = exp_time.strftime('%B') + ' ' + exp_time.stftime('%Y')
    if time_online:
        insight = f"The website {conj} been online from {reg_time} to {exp_time}"
    elif time_online:
            insight = 'No website available'
    return insight

@app.callback(Output('insight-3', 'children'),
             Input('business-name', 'value'))
def website_online(business):
    
    number_addresses = ''
    # business_df = df.query('BusinessName == @business')
    # domain_length = business_df.iloc[-1, 'time_online']
    if time_online:
        insight = f"The website has been online: {time_online}"
    if time_online:
        insight = 'No website available'
    return insight


def calculate_scores(business):
    business = 'Time Education Inc'
    toy3 = {'BusinessName':"Time Education Inc", 'num_posting' : '0'}
    toy1 = {'BusinessName':"Time Education Inc", 'url' : 'www.google.ca'}
    toy2 = {'url' : 'www.google.ca', 'register_date' : '1999-01-02', 'expire_date': '2022-09-29'}
    url_df = pd.DataFrame(toy1, index=[1])
    url_info = pd.DataFrame(toy2, index=[1])
    jb_emp_post = pd.DataFrame(toy3, index=[1])
    filtered_url_df = url_df.query('BusinessName == @business')
    # has registered website
    # longevity of website
    # has more than 1 employee
    # number of missing inputs 

    ###############################
    # length of time online 

    if url_df.loc[1,'url'] == url_df.loc[1,'url']:  # checks if nan 
        url_color = 'green'
    else: 
        url_color = 'red'
    
    url = url_df.loc[1, 'url']
    url_time_df =url_info.query('url == @url')
    url_time_df = url_time_df.set_index('url')
    reg_time = pd.to_datetime(url_time_df.loc[url, 'register_date'])
    exp_time = pd.to_datetime(url_time_df.loc[url, 'expire_date'])
    time_diff = exp_time - reg_time

    if time_diff < timedelta(7) or url_color == 'red':
        longevity_color = 'red'
    elif time_diff < timedelta(365):
        longevity_color = 'yellow'
    else:
        longevity_color='green'

    ##############################
    # Job Posting

    num_posting = jb_emp_post.query('BusinessName == @business').iloc[0,1]

    if int(num_posting) > 0:
        job_post_color = 'green'
    else:
        job_post_color = 'red'

    #############################

    scores = [url_color, longevity_color, job_post_color, 'red']
    return scores

@app.callback(Output('pie-chart', 'figure'),
             [Input('feature_type', 'value'),
            Input('business-name', 'value')])
def plot_donut(score, business):

    score_list = calculate_scores(business)
    df_dict = {'feat': ['website', 'reviews', 'government', 'other'],
           'size': [25, 25 ,25,25],
           'score' : score_list}

    pie_df = pd.DataFrame(df_dict)

    fig = go.Figure(data=[go.Pie(labels=pie_df['feat'],
                             values=[25,25,25,25])])
    fig.update_traces(hoverinfo='label+percent', textinfo='label', textfont_size=20,
                    marker=dict(colors=pie_df['score'], line=dict(color='#000000', width=1)))
    fig.update_layout(showlegend=False)

    return fig

@app.callback(Output('address', 'children'),
             [Input('business-name', 'value')])
def update_address(business):
    
    business_df = df.query('BusinessName == @business')
    if business_df.iloc[-1, 13] == business_df.iloc[-1, 13]:
        house = str(int(business_df.iloc[-1, 13]))
    else: 
        house = ''
    if business_df.iloc[-1, 14] == business_df.iloc[-1, 14]:
        street = str(business_df.iloc[-1, 14])
    else: 
        street = ''
        
    if business_df.iloc[-1, 11] == business_df.iloc[-1, 11]:
        house = business_df.iloc[-1, 11] + " " + house
    
    return  house + ' ' + street

@app.callback(Output("histogram", "figure"),
             [Input('feature_type', 'value'),
             Input('business-name', 'value'),
             Input('std', 'value')])
def plot_hist(xaxis, business,sd):
    print(business_info.head(5))
    xrange = None
    ci_color = 'black'
    business_df = df.query('BusinessName == @business')
    type_value = business_df.iloc[0, 9]
    business_join = pd.merge(business_info, df, how='left', on='BusinessName')
    print(business_join.head())
    if sd == '':
        sd = 1

    if xaxis == 'Fee Paid':
        clean_name = 'Fees Paid'
        xaxis = 'FeePaid by'
        index = -3
        estimate = business_df.iloc[-1, index] # use -4 for employees, -3 for FeesPaid
        hist_data = df.query('BusinessType == @type_value').loc[:, xaxis]
        lower_thresh, upper_thresh, _ = within_thresh(estimate, type_value, xaxis, df, sd)
        xrange=[0, upper_thresh*1.25]

    elif xaxis == 'Number of Employees':
        clean_name = 'Number of Employees at'
        xaxis = 'NumberofEmployees'
        index = -4
        estimate = business_df.iloc[-1, index]
        hist_data = df.query('BusinessType == @type_value').loc[:, xaxis]
        lower_thresh, upper_thresh, _ = within_thresh(estimate, type_value, xaxis, df, sd)
        xrange=[0, upper_thresh*1.25]
        
    elif xaxis == 'Total Fees Paid':
        clean_name = 'Total Fees Paid by'
        estimate = df.groupby('BusinessName').sum().loc[business, "FeePaid"]
        hist_data = df.groupby('BusinessName').sum().loc[:, 'FeePaid']
        lower_thresh, upper_thresh, _ = within_thresh(estimate, type_value, xaxis, df, sd)
        xaxis = 'FeePaid'
        xrange=[0, upper_thresh*1.25]
    
   # elif xaxis == 'turnover':
        # business_info.query('BusinessName == @business')
        # estimate = business_df.iloc[-1, 2]
        # hist_data = df.query('BusinessType == @type_value')
       #estimate = business_join.groupby('BusinessName')

    elif xaxis == 'Missing Values':
        clean_name = 'Missing Values in'
        business_df['Average Number of Missing Values'] = business_df.isnull().sum(axis=1)
        estimate = business_df.loc[:,'Average Number of Missing Values'].sum()/business_df.loc[:,'Average Number of Missing Values'].mean()
        missing_df = pd.DataFrame.copy(df)
        missing_df['Average Number of Missing Values'] = df.isnull().sum(axis=1)
        total_missing = missing_df.groupby('BusinessName').sum().loc[:,'Average Number of Missing Values']
        count_reports = missing_df.groupby('BusinessName').count().loc[:,'Average Number of Missing Values']
        hist_data = total_missing/count_reports
        xaxis = 'Average Number of Missing Values'
        lower_thresh = estimate
        upper_thresh = estimate
        ci_color = 'red'


    fig = px.histogram(hist_data, x = xaxis,height=400)
    fig.update_xaxes(range=xrange)
    fig.update_layout(shapes=[
        dict(
        type= 'line',
        line_color = ci_color,
        yref= 'paper', y0= 0, y1= 1,
        xref= 'x', x0= lower_thresh, x1= lower_thresh
        ),
        dict(
        type= 'line',
        line_color = ci_color,
        yref= 'paper', y0= 0, y1= 1,
        xref= 'x', x0= upper_thresh, x1=upper_thresh
        ),
        dict(
        type= 'line',
        line_color='red',
        yref= 'paper', y0= 0, y1= 1,
        xref= 'x', x0= estimate, x1=estimate
        ),
    ], title=f"Distribution of {clean_name} {type_value} companies"
    )
    return fig

@app.callback(Output("score", "children"),
             [Input('feature_type', 'value'),
             Input('business-name', 'value'),
             Input('std', 'value')])
def count_sigs(xaxis, business,sd):
    business_df = df.query('BusinessName == @business')
    type_value = business_df.iloc[0, 9]
    featlist = ['FeePaid', 'NumberofEmployees', 'Total Fees Paid']

    sd = 1

    for feat in featlist:

        if feat == 'FeePaid':
            index = -3
            estimate = business_df.iloc[-1, index] # use -4 for employees, -3 for FeesPaid
            hist_data = df.query('BusinessType == @type_value').loc[:, feat]
            print(feat)
            _, __, containsFee = within_thresh(estimate, type_value, feat, df, sd)

        elif feat == 'NumberofEmployees':
            index = -4
            estimate = business_df.iloc[-1, index]
            hist_data = df.query('BusinessType == @type_value').loc[:, feat]
            _, __, containsEmp= within_thresh(estimate, type_value, feat, df, sd)
            
        elif feat == 'Total Fees Paid':
            estimate = df.groupby('BusinessName').sum().loc[business, "FeePaid"]
            hist_data = df.groupby('BusinessName').sum().loc[:, 'FeePaid']
            _, __, containstotfee = within_thresh(estimate, type_value, feat, df, sd)

    sum_score = containsFee + containsEmp + containstotfee
    output = len(featlist) - sum_score
    return output




# last year that they gave info to public 
# number of employees
# operating revenue 
# glassdoor presence






if __name__ == '__main__':
    app.run_server(debug=True)