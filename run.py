
""" TEST PROGRAMS WE WANT TO IMPORT """

def main():

    """ PARSE ARGUMENTS FOR WHICH TEST YOU WANT TO RUN """
    parser = argparse.ArgumentParser()

    """ CONFIGS TO OPT OUT OF AUTO-CONFIG SET UP """
    parser.add_argument("-results_path", "--results_path", type=str, default=None)
    parser.add_argument("-config_path", "--config_path", type=str, default=None)

    """ CONFIGS TO OPT OUT OF AUTO-CONFIG SET UP """
    parser.add_argument("-setting", "--setting", type=str, default='control')
    parser.add_argument("-test", "--test", type=str, default='test_1')

    """ USE SETTINGS TO CALL YOUR DESIRED TESTS """
    for

if __name__ == '__main__':
    main()
