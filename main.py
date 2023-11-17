import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 


import uvicorn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

dataframe = pd.read_csv("./BANGALORE.csv")

dataframe_copy1 =dataframe.drop(columns=['_id', 'Date Time',  'company', 'Location', 'Start Date',
       'Stipend', 'Duration', 'Apply by Date', 'Offer']).copy()

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(dataframe_copy1["Skills and Perks"])
# print(matrix.shape)
cosine_sim = cosine_similarity(matrix , matrix)

def get_intership(skills):
    selected_skills = ', '.join(skills)

    new_vector = vectorizer.transform([selected_skills])

    cosine_sim_with_selected_skills = cosine_similarity(new_vector, matrix)

    sim_scores = list(enumerate(cosine_sim_with_selected_skills[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    indices = [score[0] for score in sim_scores[:5]]
    recommendations = []
    for i in indices:
       recommendations.append({"Internship" : dataframe['profile'].iloc[i] , "Company" : dataframe["company"].iloc[i] , "Stipend" : dataframe["Stipend"].iloc[i] , "Location" : dataframe["Location"].iloc[i] })
    
    return recommendations

app = FastAPI()


origins = [
    "https://workshala-in.vercel.app",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/internship/{skills}")
def recommendation_func(skills : str):
    recommendations = get_intership([skills])
    return recommendations