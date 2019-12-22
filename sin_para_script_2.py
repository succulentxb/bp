import sin_fitting as sf

if __name__ == "__main__":
    for neu_num_1 in range(10, 50, 10):
        for neu_num_2 in range(10, 100, 10):
            sizes = [1, neu_num_1, neu_num_2, 1]
            for learning_rate in range(5, 30, 5):
                train_result = sf.network_train(sizes, learning_rate / 100)
                file = open("./para_test_2", "a+")
                file.write("---------------------------\n")
                file.write("sizes: " + str(sizes) + ", learning rate: " + str(learning_rate / 100) + "\n")
                file.write("train error: " + str(train_result['error']) + "\n")
                file.write("---------------------------\n")
                file.close()