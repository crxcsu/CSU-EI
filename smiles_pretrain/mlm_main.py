import warnings, os
warnings.filterwarnings("ignore")
from smiles_mlm import main


if __name__ == '__main__':
    MAIN_PATH = os.getcwd()
    main(MAIN_PATH)




