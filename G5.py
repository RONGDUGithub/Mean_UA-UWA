import winsound
from mean_multiple_services import *
import time


def run_experiment(file_name, mechanisms, epsilons):
    # data_array = pd.read_excel(file_name)
    # data_array = pd.read_csv(file_name, header=None, sep='\n')
    # data0 = data_array.to_numpy()
    # data0 = data0.flatten()

    try:
        # 读取文件
        data_array = pd.read_csv(file_name, header=None, sep='\n')

        # 转换为numpy数组
        data0 = data_array.to_numpy()

        # 如果数组是多维的，将其展平
        if data0.ndim > 1:
            data0 = data0.flatten()

        # 确保数据类型是float
        data0 = data0.astype(float)

    except Exception as e:
        print(f"读取文件出错: {e}")
        # 可以在这里添加备选的读取方法
        try:
            with open(file_name, 'r') as file:
                data0 = np.array([float(line.strip()) for line in file])
        except Exception as e:
            print(f"备选读取方法也失败: {e}")
    max_val = np.max(data0)
    min_val = np.min(data0)
    data0 = 2 * (data0 - min_val) / (max_val - min_val) - 1
    mean_ground_truth = np.mean(data0)
    user_number = len(data0)

    min_data0 = np.min(data0)
    max_data0 = np.max(data0)

    # 重复运行的次数
    num_runs = 20  # 10
    result_run = np.zeros(len(mechanisms)+2)
    time_sta = 0
    for run in range(num_runs):

        d_default = 100
        results, nor_noise, mean_list = multi_mechanism_process(mechanisms, data0, min_data0, max_data0, epsilons)

        midpoints_minus1_to_1 = torch.tensor(np.linspace(-1, 1 - 2 / d_default, d_default) + 1 / d_default).cuda()
        batch_size = user_number // 100
        final_weight_sum = 0

        start_time = time.time()
        for i in tqdm(range(1, int(user_number / batch_size + 1))):  # user_number + 1
            batch_noise = torch.vstack(results)[:, (i - 1) * batch_size: i * batch_size]
            batch_nor = torch.vstack(nor_noise)[:, (i - 1) * batch_size: i * batch_size]

            mean_tensor = compute_weighted_mean1(
                mechanisms,
                epsilons,
                batch_size,
                batch_noise,
                batch_nor,
                d_default,
                midpoints_minus1_to_1
            )
            final_weight_sum += mean_tensor

        end_time = time.time()
        execution_time = end_time - start_time
        time_sta += execution_time

        mean_tensor = final_weight_sum / (user_number)
        mean_aggre = mean_tensor.detach().cpu().numpy()  # .to('cpu')
        mean_baseline = np.mean(mean_list)
        mean_list.append(mean_baseline)
        mean_list.append(mean_aggre)
        mean_list2 = np.array(mean_list, dtype=float)
        MSE_aggre = ((mean_list2 - mean_ground_truth) ** 2)
        # print('MSE_aggre', MSE_aggre)
        result_run += MSE_aggre

    time_final = time_sta / num_runs
    MSE_final = result_run/num_runs
    return MSE_final, time_final


def run_single_experiment(file_name):
    # 固定的mechanisms顺序
    mechanisms = ['duchi', 'laplace', 'piecewise', 'sw']

    # 固定的epsilon值为0.1
    epsilons_list = [
        [0.1, 0.1, 0.1, 0.1]
    ]

    # 初始化结果矩阵
    MSE_matrix = []
    time_matrix = []

    # 运行实验
    for epsilons in epsilons_list:
        MSE_final, time_final = run_experiment(file_name, mechanisms, epsilons)
        MSE_matrix.append(MSE_final)
        time_matrix.append(time_final)

    # 转换为numpy数组
    MSE_matrix = np.array(MSE_matrix)

    # 保存结果到文件
    base_name = os.path.splitext(file_name)[0]
    output_filename = f'./group5/Group5_{base_name}_A.txt'

    with open(output_filename, 'w') as f:
        f.write('Mechanisms:\n')
        f.write(str(mechanisms) + '\n')
        f.write('Epsilons:\n')
        f.write(str(epsilons_list) + '\n')
        f.write('time_matrix:\n')
        f.write(str(time_matrix) + '\n')
        f.write('MSE:\n')
        f.write(str(MSE_matrix) + '\n')

    # return MSE_matrix, time_matrix, mechanisms, epsilons_list

file_name = 'Betawave_500000.txt'
run_single_experiment(file_name)
file_name = 'Betawave_1000000.txt'
run_single_experiment(file_name)
file_name = 'Betawave_5000000.txt'
run_single_experiment(file_name)
file_name = 'Betawave_10000000.txt'
run_single_experiment(file_name)

winsound.Beep(3000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒
winsound.Beep(2000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒

winsound.Beep(4000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒
winsound.Beep(3000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒
winsound.Beep(1000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒
