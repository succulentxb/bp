import network as nw
import numpy as np

def error_eva(net):
    sample_num = 100
    rand = np.random.rand(sample_num) * (np.pi * 2) - np.pi
    result = 0
    for i in range(len(rand)):
        result += abs(np.sin(rand[i]) - net.compute([rand[i]]))
    result = result / sample_num
    return result

def network_train(sizes, learning_rate):
    network = nw.Network(sizes)
    # change the biases of network, the original biases of network are -0.5
    #for i in range(len(network.biases)):
    #    network.biases[i] += (bias + 0.5)
    #network.outputs_bias += (bias + 0.5)

    print("------------------------------------------------")
    print("train begin, sizes=%s, learning_rate=%f" % (str(sizes), learning_rate))
    train_time = 0
    train_upper = 3000
    error_result = 0
    while train_time < train_upper:
        train_num = -np.pi
        while train_num < np.pi:
            network.train([train_num], [np.sin(train_num)], learning_rate)
            train_num += 0.01
        train_num = np.pi
        train_time += 1
        error_val = error_eva(network)
        print("train time: %d, error value: %f" % (train_time, error_val))
        if train_upper - train_time < 20:
            error_result += error_val
    error_result = error_result / 20
    print("train error: ", error_result)
    print("---------------------------------")
    return {'error': error_result, 'network': network}
'''
error_val = 1
while error_val > 0.1:
    rand = np.random.rand(1000) * (np.pi * 2) - np.pi
    rand_sin = np.sin(rand)
    for i in range(len(rand)):
        print("train with input %f" % (rand[i]))
        network.train([rand[i]], [rand_sin[i]], 0.3)
        print("-------------------------------")
    result = []
    tmp = 0
    for i in range(len(rand)):
        tmp += (rand_sin[i] - network.compute([rand[i]])) ** 2
    error_val = tmp / 2
'''
if __name__ == "__main__":
    ret = network_train([1, 60, 1], 0.1)
    while True:
        num_in = float(input("enter a num in [-pi, pi]:"))
        result = ret['network'].compute([num_in])
        print("input: %f, output: %f, expect output: %f" % (num_in, result, np.sin(num_in)))
