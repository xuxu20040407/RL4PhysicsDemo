import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D

# 定义黑盒函数f(x)：n维的多个不同参数和大小的正态分布的叠加
def black_box_function(x, means, sigmas, weights):
    value = 0.0
    for mean, sigma, weight in zip(means, sigmas, weights):
        value += weight * np.exp(-0.5 * np.sum(((x - mean) / sigma) ** 2))
    return value

# 自定义模型类
class CustomModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# 模型预测函数（包含归一化、预测和反归一化）
def model_fitness(model, scaler_x, scaler_y, x):
    x_scaled = scaler_x.transform(x.reshape(1, -1))  # 归一化
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    y_scaled = model(x_tensor).detach().numpy()  # 模型预测
    y = scaler_y.inverse_transform(y_scaled.reshape(-1, 1))  # 反归一化
    y = np.maximum(y, 0)
    return y[0][0]

# 遗传算法
def genetic_algorithm(n, means, sigmas, weights, model=None, use_model=False, scaler_x=None, scaler_y=None, population=None, population_size=50, max_calls=1000, cxpb=0.7, mutpb=0.2):

    best_fitness = []
    call_counts = []
    calls = 0

    while calls < max_calls:
        # 评估适应度
        fitnesses = []
        for ind in population:
            if use_model and model is not None and scaler_x is not None and scaler_y is not None:
                fitness = model_fitness(model, scaler_x, scaler_y, ind)
                calls += 1
            else:
                fitness = black_box_function(ind, means, sigmas, weights)
                calls += 1
            fitnesses.append(fitness)
        fitnesses = np.array(fitnesses)
        call_counts.append(calls)

        if len(best_fitness)==0:
            best_fitness.append(np.max(fitnesses))
        else:
            best_fitness.append(np.max(fitnesses) if np.max(fitnesses)>best_fitness[-1] else best_fitness[-1])  # 寻找最大值

        # 选择
        selected_indices = np.random.choice(population_size, population_size, p=fitnesses / fitnesses.sum())
        selected_population = population[selected_indices]

        # 交叉
        for i in range(0, population_size - 1, 2):  # 确保 i + 1 不会越界
            if np.random.random() < cxpb:
                alpha = np.random.uniform(0, 1)
                child1 = alpha * selected_population[i] + (1 - alpha) * selected_population[i + 1]
                child2 = (1 - alpha) * selected_population[i] + alpha * selected_population[i + 1]
                selected_population[i] = child1
                selected_population[i + 1] = child2

        # 变异
        for i in range(population_size):
            if np.random.random() < mutpb:
                mutation = np.random.normal(0, 1, n)
                selected_population[i] += mutation

        population = selected_population

    return call_counts, best_fitness, population  # 返回优化后的种群


# 强化学习
class ReinforcementLearning:
    def __init__(self, n, means, sigmas, weights, populationsize):
        self.n = n
        self.model = CustomModel(input_size=n, hidden_size=10, output_size=1)  # 使用自定义模型
        self.X = []
        self.y = []
        self.means = means
        self.sigmas = sigmas
        self.weights = weights
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)  # 定义优化器
        self.populationsize = populationsize

    def random_sampling(self, initial_population):
        X = initial_population
        y = np.array([black_box_function(x, self.means, self.sigmas, self.weights) for x in X])
        self.X.extend(X)
        self.y.extend(y)
        return X, y

    def train_model(self, epochs=100):
        if len(self.X) > 0 and len(self.y) > 0:
            # 归一化处理
            X_scaled = self.scaler_x.fit_transform(self.X)
            y_scaled = self.scaler_y.fit_transform(np.array(self.y).reshape(-1, 1)).flatten()

            # 转换为 PyTorch 张量
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

            # 训练模型
            self.model.train()
            for epoch in range(epochs):
                self.optimizer.zero_grad()
                outputs = self.model(X_tensor).squeeze()
                loss = nn.MSELoss()(outputs, y_tensor)
                loss.backward()
                self.optimizer.step()
                # 打印训练进度（可选）
                if (epoch + 1) % 50 == 0:
                    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    def optimize(self, max_calls, top_k=5, initial_population=None, convergence_threshold=1e-4, convergence_patience=10):
        call_counts = []
        best_fitness = []
        total_calls = 0
        population = initial_population  # 使用主函数中生成的初始种群

        # 初始随机采样
        X, y = self.random_sampling(initial_population)
        total_calls += len(y)
        call_counts.append(total_calls)
        best_fitness.append(max(y))

        self.train_model()

        # 用于收敛判断的变量
        best_fitness_history = [best_fitness[-1]]
        no_improvement_count = 0

        while total_calls < max_calls:
            # 调用遗传算法函数，使用模型预测
            ga_call_counts, ga_best_fitness, population = genetic_algorithm(
                self.n, self.means, self.sigmas, self.weights,
                model=self.model, use_model=True,
                scaler_x=self.scaler_x, scaler_y=self.scaler_y,
                population=population, population_size=self.populationsize, max_calls=max_calls - total_calls,
                cxpb=0.7, mutpb=0.2
            )

            # 获取最优的前几个解
            top_indices = np.argsort(ga_best_fitness[-1])[-top_k:]
            for idx in top_indices:
                best_x = population[idx]
                best_y = black_box_function(best_x, self.means, self.sigmas, self.weights)
                self.X.append(best_x.tolist())
                self.y.append(best_y)
                total_calls += top_k

            call_counts.append(total_calls)
            best_fitness.append(max(best_fitness[-1], np.max(self.y)))

            # 收敛判断
            current_best = best_fitness[-1]
            if abs(current_best - best_fitness_history[-1]) < convergence_threshold:
                no_improvement_count += 1
            else:
                no_improvement_count = 0

            best_fitness_history.append(current_best)

            if no_improvement_count >= convergence_patience:
                print(f"Early stopping due to convergence after {total_calls} calls.")
                break

            self.train_model()

        return call_counts, best_fitness


def plot_2d_surface(means, sigmas, weights, model, scaler_x, scaler_y, title="2D Surface Plot"):
    # 初始网格划分
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z_true = np.zeros_like(X)
    Z_pred = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            Z_true[i, j] = black_box_function(point, means, sigmas, weights)
            Z_pred[i, j] = model_fitness(model, scaler_x, scaler_y, point)

    # 找到初始最大值位置
    max_index = np.unravel_index(np.argmax(Z_true, axis=None), Z_true.shape)
    max_x, max_y = X[max_index], Y[max_index]

    # 在最大值周围进行更细致的网格划分
    local_x = np.linspace(max_x - 1, max_x + 1, 50)
    local_y = np.linspace(max_y - 1, max_y + 1, 50)
    local_X, local_Y = np.meshgrid(local_x, local_y)
    local_Z_true = np.zeros_like(local_X)
    local_Z_pred = np.zeros_like(local_X)

    for i in range(local_X.shape[0]):
        for j in range(local_X.shape[1]):
            point = np.array([local_X[i, j], local_Y[i, j]])
            local_Z_true[i, j] = black_box_function(point, means, sigmas, weights)
            local_Z_pred[i, j] = model_fitness(model, scaler_x, scaler_y, point)

    # 找到更精确的最大值位置
    local_max_index = np.unravel_index(np.argmax(local_Z_true, axis=None), local_Z_true.shape)
    local_max_x, local_max_y = local_X[local_max_index], local_Y[local_max_index]
    local_max_z_true = local_Z_true[local_max_index]
    local_max_z_pred = local_Z_pred[local_max_index]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # 绘制真实函数的三维曲面图
    surf1 = ax.plot_surface(X, Y, Z_true, cmap='viridis', edgecolor='none', alpha=0.6, label='True Function')

    # 绘制更精确的最大值位置的红色误差棒
    ax.plot([local_max_x, local_max_x], [local_max_y, local_max_y], [local_max_z_true, local_max_z_pred], 'r-', linewidth=2, label='Error')

    ax.set_title('True Function vs Model Prediction')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x)')
    ax.legend()

    fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=5)
    plt.suptitle(title)
    plt.savefig("2D surface")

    return local_max_z_true


def plot_1d_curve(means, sigmas, weights, model, scaler_x, scaler_y, title="1D Curve Plot"):
    # 初始网格划分
    x_range = np.linspace(-10, 10, 400)
    y_true = np.array([black_box_function(x, means, sigmas, weights) for x in x_range])
    y_pred = np.array([model_fitness(model, scaler_x, scaler_y, x) for x in x_range])

    # 找到初始最大值位置
    max_index = np.argmax(y_true)
    max_x = x_range[max_index]

    # 在最大值周围进行更细致的网格划分
    local_x_range = np.linspace(max_x - 1, max_x + 1, 100)
    local_y_true = np.array([black_box_function(x, means, sigmas, weights) for x in local_x_range])
    local_y_pred = np.array([model_fitness(model, scaler_x, scaler_y, x) for x in local_x_range])

    # 找到更精确的最大值位置
    local_max_index = np.argmax(local_y_true)
    local_max_x = local_x_range[local_max_index]
    local_max_y_true = local_y_true[local_max_index]
    local_max_y_pred = local_y_pred[local_max_index]

    # 绘制曲线
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, y_true, label='True Black-Box Function', color='blue')
    plt.plot(x_range, y_pred, label='Model Prediction', color='red', linestyle='--')

    # 绘制最大值位置的红色误差棒
    plt.vlines(local_max_x, local_max_y_true, local_max_y_pred, colors='r', linestyles='-', linewidth=2, label='Error')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig("1D curve")

    return local_max_y_true


def plot_higher_dimensional_max(means, sigmas, weights, model, scaler_x, scaler_y, n,
                                title="Higher Dimensional Max Search"):
    # 初始搜索范围
    initial_bounds = [(-10, 10)] * n
    # 迭代次数和每次迭代的网格密度
    iterations = 5
    grid_density = [30, 20, 20, 10, 10]  # 每次迭代的网格密度逐渐增加

    current_bounds = initial_bounds
    best_point = None
    best_value = -np.inf

    for iter in range(iterations):
        # 根据当前边界生成网格点
        grid = []
        for i in range(n):
            grid.append(np.linspace(current_bounds[i][0], current_bounds[i][1], grid_density[iter]))
        meshgrid = np.meshgrid(*grid)

        # 将网格点转换为可以输入模型的格式
        points = np.vstack([meshgrid[i].flatten() for i in range(n)]).T

        # 计算每个点的真实函数值和模型预测值
        true_values = np.array([black_box_function(point, means, sigmas, weights) for point in points])
        pred_values = np.array([model_fitness(model, scaler_x, scaler_y, point) for point in points])

        # 找到当前网格中的最大值点（真实值）
        max_index = np.argmax(true_values)
        current_best_point = points[max_index]
        current_best_value = true_values[max_index]

        # 更新全局最大值
        if current_best_value > best_value:
            best_value = current_best_value
            best_point = current_best_point

        # 缩小搜索范围，围绕当前最佳点
        new_bounds = []
        for i in range(n):
            center = best_point[i]
            width = current_bounds[i][1] - current_bounds[i][0]
            new_lower = max(current_bounds[i][0], center - width * 0.1)  # 缩小10%
            new_upper = min(current_bounds[i][1], center + width * 0.1)
            new_bounds.append((new_lower, new_upper))
        current_bounds = new_bounds

        print(f"Iteration {iter + 1}: Best value found {best_value} at {best_point}")

    # 可视化结果（由于维度较高，这里仅打印结果）
    print(f"\nFinal best value: {best_value} at point {best_point}")

    # 返回最大值和对应点
    return best_value, best_point

# 主函数
def main():
    # 用户给定n值
    n = 3  # 可以设置为1或2
    max_calls_n = [100,200,2000]
    max_calls = max_calls_n[n-1]
    top_k_n=[1,1,1]
    top_k=top_k_n[n-1]
    populationsize_n=[10,100,1000]
    population_size=populationsize_n[n-1]

    # 定义多个正态分布的参数
    num_distributions = 5
    means = np.random.uniform(-10, 10, (num_distributions, n))
    sigmas = np.random.uniform(1, 3, (num_distributions, n))
    weights = np.random.uniform(0.5, 1, num_distributions)

    # 在主函数中生成初始种群
    if n == 1:
        initial_population = np.linspace(-10, 10, population_size).reshape(-1, 1)
    else:
        grid = np.linspace(-10, 10, int(population_size ** (1 / n)))
        initial_population = np.array(np.meshgrid(*[grid] * n)).T.reshape(-1, n)
    population_size=initial_population.shape[0]
    # 运行遗传算法（使用黑盒函数）
    ga_call_counts, ga_best_fitness, _ = genetic_algorithm(n, means, sigmas, weights, population=initial_population,population_size=population_size, max_calls=max_calls, cxpb=0.7, mutpb=0.2)

    # 运行强化学习（使用模型预测）
    rl = ReinforcementLearning(n, means, sigmas, weights,population_size)
    rl_call_counts, rl_best_fitness = rl.optimize(
        max_calls,
        top_k=top_k,
        initial_population=initial_population, convergence_threshold=1e-4, convergence_patience=500
    )

    # 根据维度绘制曲线或等高线图
    if n == 1:
        ymax=plot_1d_curve(means, sigmas, weights, rl.model, rl.scaler_x, rl.scaler_y, title="1D Curve Plot")
    elif n == 2:
        ymax=plot_2d_surface(means, sigmas, weights, rl.model, rl.scaler_x, rl.scaler_y, title="2D Contour Plot")
    else:
        ymax, best_point = plot_higher_dimensional_max(means, sigmas, weights, rl.model, rl.scaler_x, rl.scaler_y, n, title="Higher Dimensional Max Search")
    # 绘制对比曲线
    plt.figure(figsize=(10, 6))
    plt.plot(ga_call_counts, ga_best_fitness, label='Genetic Algorithm')
    plt.plot(rl_call_counts, rl_best_fitness, label='Reinforcement Learning')

    plt.plot([0, max_calls], [ymax, ymax], label='Black Box')

    plt.xlabel('Number of Black-Box Function Calls')
    plt.ylabel('Best Fitness')
    plt.title('Comparison of Optimization Algorithms')
    plt.legend()
    plt.grid()
    plt.savefig("Comparison of "+str(n)+" dimension(s)")

if __name__ == '__main__':
    main()