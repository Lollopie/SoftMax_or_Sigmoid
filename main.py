import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pos_softMax', type=int)
    parser.add_argument('pos_modelIndex', type=int)
    parser.add_argument('pos_datasetIndex', type=int)
    parser.add_argument('pos_calibrationMode', type=int, required=False)
    args = parser.parse_args()
