import os
import shutil

def rename_emails(dataset, index_file, save_dir):
    with open(index_file) as index_fp:
        index = index_fp.read().replace('../data/', '').splitlines()
        for target_file in index:
            email_attrs = target_file.split() # Email attributes [type, filename] (type = ham or spam)
            shutil.copy2(dataset + email_attrs[1], save_dir + email_attrs[1] + '.' + email_attrs[0])
            print("File " , email_attrs[1], "Copied")