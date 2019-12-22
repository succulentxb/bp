import sin_fitting as sf 

if __name__ == "__main__":
    for neu_num in range(10, 310, 10):
        sizes = [1, neu_num, 1]
        for learning_rate in range(5, 50, 5):
            train_result = sf.network_train(sizes, learning_rate / 100)
            file = open("./para_test_1", "a+")
            file.write("---------------------------\n")
            file.write("sizes: " + str(sizes) + ", learning rate: " + str(learning_rate / 100) + "\n")
            file.write("train error: " + str(train_result['error']) + "\n")
            file.write("---------------------------\n")
            file.close()