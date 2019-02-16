import utils.rename_email as util

def main():
    dataset_dir = 'dataset/trec07p/'
    index_file = dataset_dir + 'full/index'

    util.rename_emails(dataset_dir + 'data/', index_file, 'dataset/trec07p_clean/data/')

if __name__ == "__main__":
    main()