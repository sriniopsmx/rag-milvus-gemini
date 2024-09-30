#!/usr/bin/env python
# coding: utf-8

# ### Program to Get Context for an Issue
# End result is a Query that can be send to an LLM

# In[55]:


# Mivus
from pymilvus import MilvusClient, model,connections, db

# Huggingface
from datasets import Dataset, load_dataset # For loading
from transformers import AutoTokenizer, AutoModel

# Pytorch
import torch
import pandas as pd
import google.generativeai as genai
#import GoogleGenerativeAI, HarmCategory, HarmBlockThreshold 
#import PIL.Image
import os
import streamlit as st 
import hmac


# In[75]:


milHost=os.environ["MIL_HOST"]
milURI=os.environ["MIL_URI"]
milDBname=os.environ["MIL_DB_NAME"]
milCollection=os.environ["MIL_COLLECTION"]
modelDimension= os.environ["MIL_MODEL_DIM"]

#milHost="192.168.1.44"
#milURI='http://192.168.1.44:19530'
#milDBname="milvus_demo"
#milCollection="demo_collection"
#modelDimension= 384 # 768
activityLogFile= os.environ["ACTIVITY_LOG_FILE"]
google_API_key= os.environ["GOOGLE_API_KEY"]
geminModel="gemini-1.5-flash"


# In[76]:


generationConfig = {
	"temperature": 0,
}


# In[77]:


genai.configure(api_key= google_API_key)
model = genai.GenerativeModel(geminModel, system_instruction="You an expert in Open Source Spinnaker. Please answer with the possible causes and resolutions to the issue in 5 to 10 lines.")


# In[78]:


# This is model being used for RAG, not for inference
MODEL =  "sentence-transformers/all-MiniLM-L6-v2"  # Name of model from HuggingFace Models
INFERENCE_BATCH_SIZE = 64  # Batch size of model inference, may be used for uploading the docs into Milvus?

# Load tokenizer & model from HuggingFace Hub
ragTokenizer = AutoTokenizer.from_pretrained(MODEL) #This will be used to tokenize the inputs and uploading into Milvus
ragModel = AutoModel.from_pretrained(MODEL) # This should be used for finding relavent docs
ragData = pd.DataFrame(columns=['Relevance', 'Issue','Context'])
ragPrompt="Please wait"
changeStr=". Nothing has apparently changed."
changeHappened = False
answer = ""


# In[79]:


# I am tokenizing only the incident description
# data.map uses this function to 
def encode_text(batch):
    # Tokenize sentences
    encoded_input = ragTokenizer(
        batch["Incident Description"], padding=True, truncation=True, return_tensors="pt"
    )

    # Compute token embeddings
    with torch.no_grad():
        model_output = ragModel(**encoded_input)

    # Perform pooling
    token_embeddings = model_output[0]
    attention_mask = encoded_input["attention_mask"]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sentence_embeddings = torch.sum(
        token_embeddings * input_mask_expanded, 1
    ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Normalize embeddings
    batch["description_embedding"] = torch.nn.functional.normalize(
        sentence_embeddings, p=2, dim=1
    )
    return batch


# In[80]:


def get_response(basePrompt, changeHappened):
#basePrompt= "Spinnaker deployments are failing with access denied. No configurations have changed since it was last successful."
    questions = {
        "Incident Description": [
            # "account access denied",
            # "deployments are very slow"
            # "sky is falling",
           basePrompt,
        ]
    }

    # Generate question embeddings
    question_embeddings = [v.tolist() for v in encode_text(questions)["description_embedding"]]
    milvus_client = MilvusClient(milURI)
    # Now we search for our Incident description in Milvus
    search_results = milvus_client.search(
        collection_name=milCollection,
        data=question_embeddings,
        limit=5,  # How many search results to output
        output_fields=["Responses", "Context"],  # Include these fields in search results
    )

    # Prepare prompt info from RAG DB
    ragInfo=""
    for q, res in zip(questions["Incident Description"], search_results):
        for r in res:
            ans= r["entity"]["Responses"]
            ctxt = r["entity"]["Context"]
            score = r["distance"]
            #ragData.add({'Relevance':ans, 'Issue':ctxt,'Context':score})
            ragData.loc[len(ragData.index)] = [ score, ans, ctxt ] 
            if score > 0.6:
                #print(f"{ans}\n") 
                ragInfo += ans + "\n"
                if len(ctxt) > 10 :
                    ragInfo += ctxt + "\n"

    
    # Prepare PROMPT
    if changeHappened :
       ragPrompt="A user is facing this issue:" + basePrompt + " Here are some responses to similar issues:"
    else:    
       ragPrompt="A user is facing this issue:" + basePrompt + changeStr + " Here are some responses to similar issues:"

    ragPrompt += ragInfo
    
    if len(ragInfo) < 10 :
        return "**There was'nt enough information in current experience DB to get customer specific response.Please ensure that the DB is updated and reloaded**", ragPrompt
    return model.generate_content(ragPrompt, generation_config=generationConfig).text, ragPrompt


# In[81]:


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# Main Streamlit app starts here
st.set_page_config(layout="wide")
#st.write("Here goes your normal Streamlit app...")


# In[82]:


#response, ragPrompt = get_response("spinnaker deployment giving 404", changeHappened)
#response, ragPrompt = get_response("the sky turned white", changeHappened)
#ragPrompt, response


# In[70]:





# In[40]:


colA, colB, colC = st.columns([.50, .10, .40])
response=""
with colA:
    basePrompt = st.text_input("Please enter the user issue, ensure that 'spinnaker' word is present:", value="", key="prompt")
with colB:
    changeHappened = st.checkbox("Something changed?", value=False, key="change")
    if st.button("Press", key="button"):
        response, ragPrompt = get_response(basePrompt, changeHappened)
        print("RAGPrompt:", ragPrompt)
        answer = response
with colA:
    st.markdown(answer)


# In[41]:


# df.append({'Relevance':0.6, 'Issue':'hello','Context':'newinfo'})
with colC:
  st.subheader("Related Items found (top 3 with score > 0.6 are considered)")
  st.table(ragData)
  st.write("Final Prompt:" + ragPrompt)


# In[85]:


with open(activityLogFile, "a") as outFile:
    # Writing data to a file
    if len(answer) > 0 :
     answerInOneLine = answer.replace("\n", "")
     answerInOneLine = answerInOneLine.replace("*", "")
     answerInOneLine = answerInOneLine.replace(",", ";") # Replace comma with semi-colon so CSV format is retained
     outFile.write(basePrompt +"," + answerInOneLine +"\n")


# In[86]:


# for q, res in zip(questions["Incident Description"], search_results):
#     print("Incident Description:", q)
    
#     for r in res:
#         ans= r["entity"]["Responses"]
#         ctxt = r["entity"]["Context"]
#         score = r["distance"]
#         if score > 0.4:
#             print ("Here are the selections\n")
#             print(f"Ans: {ans}\n") 
#             print(f"\tContext: {ctxt}\n")
#             print(f"\tScore: {score}\n")


# In[91]:





# In[11]:


# print(ragPrompt)
# resp = get_response(ragPrompt).text


# In[ ]:




