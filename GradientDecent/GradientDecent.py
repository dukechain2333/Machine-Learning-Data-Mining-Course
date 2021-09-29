import matplotlib.pyplot as plt


def origin_func(x):
    return x**2+2*x+1


def derivatives_func(x):
    return x*2+2


def gradient_decent(cur_x, learning_rate, epoch, percision, derivate):
    data_x = []
    for i in range(epoch):
        grad = derivate(cur_x)
        data_x.append(cur_x)
        if grad < percision:
            break
        cur_x = cur_x - learning_rate*grad
        y = origin_func(cur_x)
        print(
            f'This is epoch: {i+1}, current x is {cur_x} ,current value is {y}')

    print(f"Min Value is {y}")

    return data_x


if __name__ == '__main__':
    cur_x = 10
    learning_rate = 0.01
    epoch = 100000
    percision = 0.00001
    data_x = gradient_decent(cur_x=cur_x, learning_rate=learning_rate,
                             epoch=epoch, percision=percision
                            , derivate=derivatives_func)
    x = range(-10, 11)
    y = [origin_func(i) for i in x]
    y_hat = [origin_func(i) for i in data_x]
    plt.plot(x, y)
    plt.scatter(data_x, y_hat, s=10, marker='*')
    plt.show()
