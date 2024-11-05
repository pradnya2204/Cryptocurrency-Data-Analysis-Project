import streamlit as st
import requests

def app():
    st.title("Cryptocurrency News")

    # Define the API URL
    api_url = "https://newsdata.io/api/1/news?apikey=pub_31232a17ff003844da601608d829737bd8380&q=Cryptocurrency&language=en"

    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        data = response.json()
        articles = data.get("results", [])

        if articles:
            for index, article in enumerate(articles):
                title = article.get("title", "")
                description = article.get("description", "")
                source = article.get("creator", "")
                published_at = article.get("pubDate", "")
                url = article.get("url", "")  # Corrected to "url"
                image_url = article.get("image_url", "")  # Ensure correct key for image URL

                col1, col2 = st.columns([1, 2])  # Split the layout into two columns

                # Display the image on the left with adjusted size if image_url is not null
                with col1:
                    if image_url:
                        st.image(image_url, caption="Image", use_column_width='auto', width=100)  # Adjusted width

                # Display title and description on the right
                with col2:
                    st.subheader(title)
                    st.write(description)
                    st.write(f"Source: {source} - Published: {published_at}")

                    # Use a unique key for each button
                    button_key = f"read_more_{index}"
                    if st.button("Read More", key=button_key):
                        st.markdown(f"[Read the full article]({url})")
                st.markdown('---')
        else:
            st.write("No news articles found.")
    except requests.RequestException as e:
        st.warning(f"An error occurred while fetching cryptocurrency news: {e}")

if __name__ == "__main__":
    app()
