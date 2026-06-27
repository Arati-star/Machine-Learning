from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

model = pickle.load(
    open("models/model.pkl","rb")
)