import streamlit as st
from streamlit_option_menu import option_menu

import home, LSTM_Algo, Upload_CSV, News, Calculator

st.set_page_config(
    page_title="Cryptocurrency",
)

class MultiApp:

    def __init__(self):
        self.apps = []
    
    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })
    
    def run(self):
        with st.sidebar:
            app = option_menu(
                menu_title="Cryptocurrency",
                options=['Home', 'LSTM', 'CSV', 'News', 'Calculator'],
                icons=['house-fill', 'robot', 'filetype-csv', 'newspaper', 'calculator'],
                menu_icon='currency-exchange',
                default_index=0,  # Set default index to 0 for "Home"
                styles={
                    "container": {"padding": "5!important", "background-color": 'black'},
                    "icon": {"color": "white", "font-size": "20px"},
                    "nav-link": {"color": "white", "font-size": "20px", "text-align": "left", "margin": "0px", "--hover-color": "blue"},
                    "nav-link-selected": {"background-color": "#02ab21"},
                }
            )

        if app == "Home":
            home.app()
        if app == "LSTM":
            LSTM_Algo.app()
        if app == "CSV":
            Upload_CSV.app()
        if app == "News":
            News.app()
        if app == "Calculator":
            Calculator.app()
    
app = MultiApp()
app.run()
