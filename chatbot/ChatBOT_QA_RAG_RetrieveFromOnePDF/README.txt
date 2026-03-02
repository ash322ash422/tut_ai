
PREREQUISITES:
- Have an open ai key

1) Install python3.10: https://www.python.org/downloads/release/python-31011/
-> Install python-3.10.11-amd64.exe
-> Check box: "Add Python 3.10 to PATH"
-> Type: py -3.10
This should work

2) Using VScode, create virtual env:
Command palette -> Create Env -> Select Interpreter
C:\Users\<user-name>\AppData\Local\Programs\Python\Python310\python.exe

3) Activate virtual environment:
>  .\.venv\Scripts\activate

OPTION1) use requirements.txt:
a) > python -m pip install -r requirements.txt

NOTE: I received error and had to reinstall pip:
   a) > python -m ensurepip --upgrade
   b) > python -m pip install --upgrade pip

OPTION2)
- I installed packages in following order:
pip install databutton langchain pypdf streamlit 
pip install -U langchain-community 
pip install faiss-cpu   
pip install openai==0.28


##################
1) - To run (Make sure you use virtual env python):
> python -m streamlit run .\app.py 
OR
> py -3.10 -m streamlit run .\app.py 

2) Goto browser and load a PDF: 

2.1) document-FDP
Ask question related with the PDF doc: What is the schedule for day 3?

2.2) document-RobertFrost
Ask it summarize: Summarize the PDF document in 2 or 3 lines.