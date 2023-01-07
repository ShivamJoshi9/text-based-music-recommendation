# Core Pkgs
import streamlit as st
# EDA Pkgs
import pandas as pd
import numpy as np
# Utils
import joblib
import pickle

pipe_lr = joblib.load(open("models/emotion_classifier.pkl", "rb"))
# Fxn
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
                       "sadness": "😔", "shame": "😳", "surprise": "😮"}


# Main Application
def main():
    st.title("Emotion Classifier App")
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        st.subheader("Home-Emotion In Text")

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        item = -1;
        if(item > 0): st.title("RECOMMENDED MUSIC")

        if submit_text:
            col1, col2 = st.columns(2)

            music_rec = pickle.load(open('models/music.pkl', 'rb'))
            music_rec = pd.DataFrame(music_rec)
            nd = music_rec.to_numpy()

            def recommendation(item):
                if (item > 0): st.title("RECOMMENDED MUSIC")
                item = item * 10;
                item = item - 10;
                for i in range(0, 10):
                    i = i + item;
                    st.text(nd[i, 2])

            # Apply Fxn Here
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)
                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction, emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))
                if(emoji_icon == "😠"): recommendation(1)
                elif(emoji_icon == "🤮"): recommendation(2)
                elif (emoji_icon == "😨😱"): recommendation(3)
                elif (emoji_icon == "🤗"):recommendation(4)
                elif (emoji_icon == "😂"): recommendation(5)
                elif (emoji_icon == "😐"): recommendation(6)
                elif (emoji_icon == "😔"): recommendation(7)
                elif (emoji_icon == "😔"): recommendation(8)
                elif (emoji_icon == "😳"): recommendation(9)
                elif (emoji_icon == "😮"): recommendation(10)

            with col2:
                st.success("Prediction Probability")
                st.write(probability)
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]



    else:
        st.subheader("About")


if __name__ == '__main__':
    main()