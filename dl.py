import os
import gdown
import tarfile

cwd = os.getcwd()

if not(os.path.isdir(os.path.join(cwd, "drinks"))):
    print("drinks dataset folder not found")
    if (os.path.isfile("drinks.tar.gz")):
        print("extracting drinks.tar.gz")
        tar = tarfile.open("drinks.tar.gz", "r:gz")
        tar.extractall()
        tar.close()
        print("finished extracting")
    else:
        print("drinks.tar.gz not found")
        print("downloading drinks.tar.gz")
        id = "1AdMbVK110IKLG7wJKhga2N2fitV1bVPA"
        output = "drinks.tar.gz"
        gdown.download(id=id, output=output, quiet=False)
        print("extracting drinks.tar.gz")
        tar = tarfile.open("drinks.tar.gz", "r:gz")
        tar.extractall()
        tar.close()
        print("finished extracting")