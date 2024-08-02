import os
from dotenv import load_dotenv
import anthropic

cur_notebook_dir = os.getcwd()
os.chdir("../")
base_dir = os.getcwd()
key_dir = os.path.join(base_dir, 'keys')
os.chdir(base_dir)
print(os.getcwd())
os.chdir(cur_notebook_dir)

print(load_dotenv(dotenv_path= os.path.join(key_dir, ".env")))

from get_namu_url import GetNamuUrl

inst1 = GetNamuUrl(
    os.getenv("GOOGLE_API_KEY")
    , os.getenv("GOOGLE_SEARCH_ENGINE")
    , rev_exclude= True
    , crucial_keyword = "")
res = inst1.get_url("ADSP")
[(r['title'], r['formattedUrl']) for r in res]
