# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
from llm_bedrock import retrieval_answer
from datetime import datetime, timedelta
import boto3

# Initialize DynamoDB for chat history


LOGGER = get_logger(__name__)
# 
min_year = 2000
max_year = 2024

options = ['ALL','Fact Sheet','Article','Letter','Research Report','Testimony','Regulatory Comment','Book','Spanish Publication','Other']
#
start_date = datetime(min_year, 1, 1)  #
end_date = datetime(max_year, 12, 31)  

def run():
  st.title("Policy Document Assistant")
  st.caption("A Digital Services Project")

  if "messages" not in st.session_state:
      st.session_state["messages"] = [{"role": "assistant", "content": "Input your query"}]

  for msg in st.session_state.messages:
      st.chat_message(msg["role"]).write(msg["content"])

  with st.sidebar:
      st.sidebar.title("Select Time Period")
      selected_years = st.sidebar.slider("Year", min_value=min_year, max_value=max_year, value=(2012, 2018), step=1, format="%d")
      st.sidebar.title("Select Document Type")
      types = st.sidebar.multiselect('Select Type:', options)

  if prompt := st.chat_input():
      if len(prompt) > 0:
          st.info("Your Query: " + prompt)
          answer,metadata = retrieval_answer(prompt,selected_years,types)
          #st.dataframe(answer)
          #st.markdown(answer)
          st.subheader('Answer:')
          st.write(answer)
          st.subheader('Document Metadata:')
          st.dataframe(metadata)
          st.write(types)

      else:
          st.error("Please enter a query.")


if __name__ == "__main__":
    run()