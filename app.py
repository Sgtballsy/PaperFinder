import streamlit as st
import torch
from sentence_transformers import util
import pickle


embeddings = pickle.load(open('C:\HighTeir\\researchpal\\embeddings.pkl', 'rb'))
sentences = pickle.load(open('C:\HighTeir\\researchpal\\sentences.pkl', 'rb'))
rec_model = pickle.load(open('C:\\HighTeir\\researchpal\\rec_model.pkl', 'rb'))

def recommendation(input_paper):
    cosine_scores = util.cos_sim(embeddings, rec_model.encode(input_paper))
    top_similar_papers = torch.topk(cosine_scores, dim=0, k=5, sorted=True)
    papers_list = [sentences[i.item()] for i in top_similar_papers.indices]
    return papers_list

st.set_page_config(page_title="ResearchPal", page_icon="📚", layout="wide")

st.markdown("""
    <style>
        @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css');

        body {
            font-family: 'Arial', sans-serif;
            font-size: 27px;  /* Increased by 1.5x (18px to 27px) */
            background-color: #222831;
            color: #EEEEEE;
            background-image: url("https://news.microsoft.com/wp-content/uploads/prod/sites/93/2017/08/brain1.png");
        }
        .nav-bar {
            background-color: #31363F;
            color: #EEEEEE;
            padding: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .nav-bar img {
            border: 2px solid black;
            border-radius: 5px;
            width: 150px;
        }
        .nav-bar h1 {
            font-size: 80px;  /* Increased by 2.5x (32px to 80px) */
            margin: 0;
            padding-left: 10px;
            text-align: center;  /* Center alignment */
            flex: 1;  /* To make it center-aligned with the other elements */
        }
        .nav-bar .nav-links {
            display: flex;
            gap: 20px;
        }
        .nav-bar .nav-links a {
            color: #EEEEEE;
            text-decoration: none;
            font-size: 27px;  /* Increased by 1.5x (18px to 27px) */
        }
        .nav-bar .nav-links a:hover {
            color: #76ABAE;
        }
        .search-section {
            background-color: #31363F;
            padding: 40px;
            text-align: center;
            border-radius: 15px;
            margin-top: 20px;
        }
        .search-section h2 {
            font-size: 42px;  /* Increased by 1.5x (28px to 42px) */
            color: #76ABAE;
            margin-bottom: 20px;
        }
        .stTextInput > div > div > input {
            border-radius: 30px !important;
            border: 2px solid #76ABAE !important;
            padding: 10px 15px !important;
            font-size: 27px !important;  /* Increased by 1.5x (18px to 27px) */
            background-color: #EEEEEE !important;
            color: #222831 !important;
            width: 100% !important;
            margin: 0 auto !important;
            display: block !important;
            height: 45px !important;
            box-sizing: border-box !important;
        }
        .stButton > button {
            border-radius: 30px !important;
            padding: 10px 15px !important;
            font-size: 27px !important;  /* Increased by 1.5x (18px to 27px) */
            background-color: #76ABAE !important;
            color: #222831 !important;
            width: 100% !important;
            height: 45px !important;
            box-sizing: border-box !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        .stButton > button:hover {
            background-color: #EEEEEE !important;
            color: #222831 !important;
        }
        .recommendations {
            margin-top: 40px;
        }
        .recommendation {
            background-color: #EEEEEE;
            color: #222831;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-size: 27px;  /* Increased by 1.5x (18px to 27px) */
        }
        .image-section {
            text-align: center;
            margin: 40px 0;
        }
        .image-section img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
        }
        .key-features {
            margin-top: 40px;
            padding: 20px;
            border-radius: 15px;
            background-color: #31363F;
        }
        .key-features h3 {
            font-size: 36px;  /* Increased by 1.5x (24px to 36px) */
            color: #76ABAE;
            margin-bottom: 20px;
            text-align: center;
        }
        .feature-tile {
            background-color: #76ABAE;
            color: #222831;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            font-size: 27px;  /* Increased by 1.5x (18px to 27px) */
            text-align: center;
        }
        .feature-tile i {
            font-size: 54px;  /* Increased by 1.5x (36px to 54px) */
            margin-bottom: 15px;
        }
        .footer {
            background-color: #222831;
            color: #EEEEEE;
            padding: 20px;
            text-align: center;
            margin-top: 40px;
            border-radius: 10px;
            font-size: 27px;  /* Increased by 1.5x (18px to 27px) */
        }
        .footer a {
            color: #76ABAE;
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }
    </style>
    """, unsafe_allow_html=True)

# Navigation bar
st.markdown(f"""
    <div class="nav-bar">
        <div>
            <h1>My Research Pal</h1>
        </div>
        <div class="nav-links">
            <a href="#home">Home</a>
            <a href="#about">About</a>
            <a href="#contact">Contact</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Search section
st.markdown("""
    <div class="search-section">
        <h2>Find the perfect papers for your research work. Right here.</h2>
    </div>
    """, unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])

with col1:
    input_paper = st.text_input("", placeholder="Enter topic/ paper title here", label_visibility="collapsed")

with col2:
    search_button = st.button("🔍 Search")

# Add abstract textbox below the search bar
abstract_input = st.text_area("Abstract", placeholder="Enter abstract here (optional)", height=150)

# Display recommendations
if search_button:
    if input_paper:
        st.markdown("<div class='recommendations'>", unsafe_allow_html=True)
        recommend_papers = recommendation(input_paper)
        for idx, paper in enumerate(recommend_papers, start=1):
            st.markdown(f"<div class='recommendation'><strong>{idx}.</strong> {paper}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a paper title to get recommendations.")

# Image section before key features
st.markdown("""
    <div class="key-features">
        <h3>Key Features</h3>
        <div class="feature-tile">
            <i class="fas fa-user"></i>
            <div>Personalized Recommendations</div>
        </div>
        <div class="feature-tile">
            <i class="fas fa-database"></i>
            <div>Diverse Database</div>
        </div>
        <div class=" feature-tile">
            <i class="fas fa-laptop"></i>
            <div>User-Friendly Interface</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer with additional information
st.markdown("""
    <div class="footer">
        Made by MyResearchPal Team | <a href="mailto:myazir777@gmail.com">Contact Us</a> | <a href="/privacy-policy">Privacy Policy</a>
    </div>
    """, unsafe_allow_html=True)
