import streamlit as st
import requests
import plotly.graph_objects as go

def app():
    st.title("Cryptocurrency List")

    # Define the API URL
    api_url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",  
        "order": "market_cap_desc",
        "per_page": 60,
        "page": 1,
        "sparkline": False,
    }

    try:
        response = requests.get(api_url, params=params)

        if response.status_code == 200:
            data = response.json()

            css = """
            .title {
                font-family: "Arial", sans-serif;
                color: #E0F0E3; /* Change color to your preference */
            }
            """

            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
            st.markdown('<h1 class="title">Cryptocurrency List</h1>', unsafe_allow_html=True)
            st.markdown("Price values are in USD.")

            for coin in data:
                # Display the information of each cryptocurrency
                coin_info = f'<img src="{coin["image"]}" style="width: 50px; height: 50px; object-fit: cover; margin-right: 10px;" />'
                coin_info += f'<span style="color: #ffffff;">{coin["name"]} ({coin["symbol"].upper()})</span>'
                coin_info += f'<span style="color: #00FF00;">  Price: ${coin["current_price"]:.2f}</span>'
                coin_info += f'<span style="color: #0075d3;">  Market Cap: ${coin["market_cap"]:.2f}</span>'
                coin_info += f'<span style="color: #ef0303;">  24h Change: {coin["price_change_percentage_24h"]:.2f}%</span>'
                st.markdown(coin_info, unsafe_allow_html=True)

                st.markdown('---')
        else:
            st.warning(f"Failed to fetch cryptocurrency data. Status code: {response.status_code}")
    except Exception as e:
        st.warning(f"An error occurred while fetching cryptocurrency data: {str(e)}")

if __name__ == "__main__":
    app()
