import bmp
import network_img as nw
import random
import numpy as np
import time

def max_index(arr):
    max = -10
    max_index = 0
    for i in range(len(arr)):
        if arr[i] > max:
            max = arr[i]
            max_index = i
    return max_index

def error_eva(net):
    precision = 0
    for _ in range(500):
        word_i = random.randint(1, 14)
        img_i = random.randint(250, 255)
        file_name = './TRAIN/' + str(word_i) + '/' + str(img_i) + '.bmp'
        inputs = np.array(bmp.parse(file_name))
        outputs = net.compute(inputs)
        if max_index(outputs) == word_i - 1:
            precision += 1
    return precision / 5

def network_train(sizes, learning_rate):
    network = nw.Network(sizes)

    train_time = 0
    train_upper = 100
    error_result = 0
    print("-------------------------------------------")
    print("train begin, size=%s, learning_rate=%f" % (str(sizes), learning_rate))
    while train_time < train_upper:
        for word_index in range(1, 15):
            expec_output = np.zeros(14)
            expec_output[word_index - 1] = 1
            for img_index in range(200):
                file_name = './TRAIN/' + str(word_index) + '/' + str(img_index) + '.bmp'
                inputs = np.array(bmp.parse(file_name))
                network.train(inputs, expec_output, learning_rate)
        train_time = train_time + 1
        error_val = error_eva(network)
        print("train time: %d, precision: %d%%" % (train_time, error_val))
        if train_upper - train_time < 20:
            error_result += error_val
    error_result = error_result / 20
    return {'error': error_result, 'network': network}

def network_train_quick(sizes, learning_rate):
    network = nw.Network(sizes)

    # read the images and store the vectors into a matrix
    imgs = []
    for word_index in range(1, 15):
        inner_imgs = []
        for img_index in range(200):
            file_name = './TRAIN/' + str(word_index) + '/' + str(img_index) + '.bmp'
            inputs = np.array(bmp.parse(file_name))
            inner_imgs.append(inputs)
        imgs.append(inner_imgs)
    
    train_time = 0
    train_upper = 1000
    error_result = 0
    print("---------------------------------------")
    print("train begin, size=%s, learning_rate=%f" % (str(sizes), learning_rate))
    while train_time < train_upper:
        for word_index in range(len(imgs)):
            expec_output = np.zeros(14)
            expec_output[word_index] = 1
            for img_index in range(len(imgs[word_index])):
                network.train(imgs[word_index][img_index], expec_output, learning_rate)
        
        train_time += 1
        error_val = error_eva(network)
        print("train time: %d, precision: %d%%" % (train_time, error_val))
        if train_upper - train_time < 20:
            error_result += error_val
    error_result = error_result / 20
    return {'error': error_result, 'network': network}

def network_train_quick_2(sizes, learning_rate):
    network = nw.Network(sizes)

    imgs = []
    for word_index in range(1, 15):
        inner_imgs = []
        for img_index in range(250):
            file_name = './TRAIN/' + str(word_index) + '/' + str(img_index) + '.bmp'
            inputs = np.array(bmp.parse(file_name))
            inner_imgs.append(inputs)
        imgs.append(inner_imgs)

    train_time = 0
    train_upper = 1000
    error_result = 0
    print("---------------------------------------")
    print("train begin, size=%s, learning_rate=%f" % (str(sizes), learning_rate))
    while train_time < train_upper:
        for img_index in range(200):
            for word_index in range(len(imgs)):
                expec_output = np.zeros(14)
                expec_output[word_index] = 1
                network.train(imgs[word_index][img_index], expec_output, learning_rate)
        train_time += 1
        if train_time % 10 == 0:
            error_val = error_eva(network)
            print("train time: %d, precision: %d%%" % (train_time, error_val))
            # if error_val > 84:
            #     return {'error': error_result, 'network': network}
        if train_upper - train_time < 20:
            error_result += error_val
    error_result = error_result / 20
    return {'error': error_result, 'network': network}

if __name__ == "__main__":
    time_start = time.time()
    train_result = network_train_quick_2([784, 20, 14], 0.01)
    time_end = time.time()
    print("========================================")
    print("train total time: %fs" % (time_end - time_start))
    
    while True:
        test_folder = input('enter test folder name: ')

        test_imgs = []
        for i in range(763):
            file_name = './' + test_folder + '/' + str(i) + '.bmp'
            test_imgs.append(np.array(bmp.parse(file_name)))

        # store test result in test_res(list)
        test_res = []
        for inputs in test_imgs:
            outputs = train_result['network'].compute(inputs)
            test_res.append(max_index(outputs) + 1)

        # write test result into file
        res_file = open('./checker/pred.txt', 'w')
        for res in test_res:
            res_file.write(str(res) + '\n')
        res_file.close()
        print('test result writing done')

        # evaluate final accuaracy
        accuracy = 0
        for i in range(len(test_res)):
            tmp = test_res[i] - 1
            if tmp == i % 14:
                accuracy += 1
        print('accuaracy: %f%%' % (accuracy/7.63))
'''
    while True:
        words = "苟利国家生死以岂因祸福避趋之"
        word_i = input("enter fold index (1-14): ")
        img_i = input("enter img index (0-255): ")
        file_name = './TRAIN/' + word_i + '/' + img_i + '.bmp'
        inputs = np.array(bmp.parse(file_name))
        outputs = train_result['network'].compute(inputs)
        print("recognition result: %s" % words[max_index(outputs)])
'''