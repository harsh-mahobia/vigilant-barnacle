import streamlit as st
from bot import BOT
import re

st.title("Youtube BOT")

st.text("you can ask question about any part of the video using just youtube video URL")


bot : BOT | None = None



if "bot" not in st.session_state:
    st.session_state.bot = None

video_url = st.text_input("Enter Video Code")
match = re.search(r"v=([^&]+)", video_url)
      

if st.button("search") and video_url is not "":
    if match :
        video_code = match.group(1)
        print(video_code) 
        with st.spinner("Model is loading") : 
            st.session_state.bot = BOT(video_code)
    else : 
        st.error("bad URL provided")

if st.session_state.bot is not None:
    search_query = st.text_input("Enter your Question")
    if st.button("Ask"):
        output = ""
        with st.spinner("answering..."):
            output = st.session_state.bot.query(search_query)
        st.write(output)

