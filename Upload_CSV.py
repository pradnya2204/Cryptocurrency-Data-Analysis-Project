import streamlit as st
import pandas as pd
import altair as alt

def app():
    st.title("CSV Data Visualization")

    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read and clean the data
        df = pd.read_csv(uploaded_file)
        df.dropna(inplace=True)  # Drop all rows with any missing values
        
        st.subheader("Filtered Data")
        st.dataframe(df)

        if df.empty:
            st.write("The CSV file contains no data after filtering.")
            return

        st.subheader("Line Graph")
        x_column = st.selectbox("Select a column for the X-axis", df.columns)
        y_column = st.selectbox("Select a column for the Y-axis", df.columns)

        # Check if columns contain numeric data
        if pd.api.types.is_numeric_dtype(df[x_column]) and pd.api.types.is_numeric_dtype(df[y_column]):
            chart = alt.Chart(df).mark_line().encode(
                x=x_column,
                y=alt.Y(y_column, title=y_column),
            ).properties(
                width=600,
                height=400,
                title=f"Line Chart for {y_column} over {x_column}",
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("Selected columns should be numeric for line chart visualization.")
    else:
        st.write("Please upload a CSV file to get started.")

if __name__ == "__main__":
    app()
