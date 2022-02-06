import os, csv, json, requests
from tqdm import tqdm 
from multiprocessing.pool import ThreadPool as Pool

from functools import partial

class FreebaseDownloader():
    def __init__(self, database_path, google_api_key ) -> None:
        with open(f'{database_path}/raw/entities.dict') as fin:
            l = csv.reader(fin, delimiter='\t')
            self.ids = [id for _,id in l]

        self.path = os.path.join(database_path, 'data')
        self.key = google_api_key
        os.makedirs(self.path, exist_ok = True)
        
    def download_all(self, workers=None):
        if workers == None:
            for id in tqdm(self.ids):
                self.download(id)
        else:
            with Pool(workers) as p:
                # print(FreebaseDownloader.download)
                ids = [(self, id) for id in self.ids]
                res = p.imap( partial(FreebaseDownloader.download, self), ids) 
                list(tqdm(res, total=len(self.ids)))
            # while True:
            #     ready = sum([r.ready() for r in res])
            #     print(ready)
                
                
            
    def download(self, id):
        path = os.path.join(self.path, f'{id[3:]}.json')
        if not os.path.exists(path):
            args = { 'key':  self.key, 'ids': id}
            url = 'https://kgsearch.googleapis.com/v1/entities:search/'
            try:
                r = json.loads(requests.get(url, args).content)['itemListElement']
                with open(path, 'w') as fout:
                    json.dump(r, fout)
            except:
                pass

DATABASE_PATH = './FB15k-237/FB15k-237/'
API_KEY = 'AIzaSyA1vYTArzSfaZLQ9ikIyh7RFmtOTos0v_k'

if __name__ == '__main__':
    import csv

    d = FreebaseDownloader(DATABASE_PATH, API_KEY)
    d.download_all(workers=10)
