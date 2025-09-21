import streamlit as st
from bot import BOT

st.title("PepperBot")
st.subheader("Your own Q&A-Bot")

uploaded_file = st.file_uploader("Choose a file", type=["txt"])

if "bot" not in st.session_state:
    st.session_state.bot = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.file_name:
        with st.spinner("Processing data... Please wait."):
            st.session_state.bot = BOT(uploaded_file.getvalue().decode("utf-8"))
            st.session_state.file_name = uploaded_file.name
        st.success("Process completed!")

    toggle_state = st.toggle("Show File", value=False)

    if uploaded_file.type == "text/plain" and toggle_state:
        string_data = uploaded_file.getvalue().decode("utf-8")
        st.subheader("File Content (Text):")
        st.code(string_data)
        st.write("File uploaded successfully!")
        st.write(f"File name: {uploaded_file.name}")
        st.write(f"File type: {uploaded_file.type}")
        st.write(f"File size: {uploaded_file.size} bytes")
    elif uploaded_file.type != "text/plain":
        st.write("File content processing not implemented for this type.")

if st.session_state.bot is not None:
    search_query = st.text_input("Enter your Question")
    if st.button("Ask"):
        output = ""
        with st.spinner("answering..."):
            output = st.session_state.bot.query(search_query)
        st.write(output)

# if st.session_state.bot is not None:
#     search_query = st.text_input("Enter your Question")
#     if st.button("Ask"):
#         if search_query.strip() != "":
#             with st.spinner("Answering..."):
#                 # Do NOT recreate the BOT here!
#                 output = st.session_state.bot.query(search_query)
#             st.write(output)