import pandas as pd

print("start loading data from data.py")

base_path = '/home/andyalyfsyah/FLUENT-Chatbot-2023/'
# knowledgebase_url = base_path + 'KnowledgeBaseFilkom.xlsx'
knowledgebase_url = base_path + 'KnowledgeBaseFilkom_simple.xlsx'
knowledgebase = pd.read_excel(knowledgebase_url)
knowledgebase_eval_url = base_path + 'KnowledgeBaseFilkom_eval.xlsx'
knowledgebase_eval = pd.read_excel(knowledgebase_eval_url)

qa_paired = knowledgebase.drop(columns=knowledgebase.columns.drop(['Pertanyaan', 'Jawaban']))
qa_paired.dropna(inplace=True)
print("finished loading data from data.py, get {} qa_paired".format(len(qa_paired)))

qa_paired_eval = knowledgebase_eval.drop(columns=knowledgebase_eval.columns.drop(['Pertanyaan', 'Jawaban']))
qa_paired_eval.dropna(inplace=True)
print("finished loading data from data.py, get {} qa_paired_eval".format(len(qa_paired_eval)))

qa_paired = qa_paired.applymap(lambda x: x.lower() if isinstance(x, str) else x)
qa_paired['Jawaban'] = qa_paired['Jawaban'].apply(lambda x: '[BOS]' + x + '[EOS]')
qa_paired = qa_paired.reset_index(drop=True)
print("added BOS and EOS token to qa_paired, sample qa_paired: {}".format(qa_paired.head(1)))
