import streamlit as st
from pipeline.pipeline import AnimeRecommendationPipeline
from dotenv import load_dotenv

st.set_page_config(page_title="Anime Recommender", page_icon=":guardsman:", layout="wide")
load_dotenv()

@st.cache_resource
def init_pipeline():
    return AnimeRecommendationPipeline()

pipeline = init_pipeline()
st.title("Anime Recommender System")
st.write("Ask a question about anime, and get recommendations!")
query = st.text_input("Enter your question:")

if query:
    with st.spinner("Processing your request..."):
        respose = pipeline.recommend(query)
        st.markdown("Recommendations:")
        st.write(respose)