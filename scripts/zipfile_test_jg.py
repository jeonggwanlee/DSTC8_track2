from zipfile import ZipFile

zipfile = './pp-metalwoz-dir/metalwoz-v1-normed.zip'

namelist = ZipFile(zipfile, 'r').namelist()


train_domains = set([f for f in ZipFile(zipfile, 'r').namelist() if f.startswith('dialogues/')])
