import winsound
from mean_multiple_services import *
import time


def run_experiment(data0, mechanisms, epsilons):
    data0 = data0.flatten()
    max_val = np.max(data0)
    min_val = np.min(data0)
    data0 = 2 * (data0 - min_val) / (max_val - min_val) - 1
    mean_ground_truth = np.mean(data0)
    user_number = len(data0)

    min_data0 = np.min(data0)
    max_data0 = np.max(data0)

    # 重复运行的次数
    num_runs = 50  # 10
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


def process_data(file_name, n_cols):
    result = sample_data(file_name.replace('.xlsx', '.csv'), n_cols)

    mechanisms = ['duchi', 'laplace', 'piecewise', 'sw']
    epsilons = [0.1, 0.1, 0.1, 0.1]

    MSE_matrix = []
    time_matrix = []

    for col in result.columns:
        data = result[col].dropna().values
        MSE_final, time_final = run_experiment(data, mechanisms, epsilons)
        MSE_matrix.append(MSE_final)
        time_matrix.append(time_final)

    MSE_matrix = np.array(MSE_matrix)
    time_matrix = np.array(time_matrix)

    MSE_avg = np.mean(MSE_matrix, axis=0)
    time_avg = np.mean(time_matrix, axis=0)

    base_name = os.path.splitext(file_name)[0]
    output_filename = f'./multiple_data/Group6_{base_name}_{n_cols}.txt'

    with open(output_filename, 'w') as f:
        f.write('Mechanisms:\n')
        f.write(str(mechanisms) + '\n')
        f.write('Epsilons:\n')
        f.write(str(epsilons) + '\n')
        f.write('time_matrix:\n')
        f.write(str(time_avg) + '\n')
        f.write('MSE:\n')
        f.write(str(MSE_avg) + '\n')

# 使用示例:
file_name = 'output4.xlsx'
n_cols = 10
process_data(file_name, n_cols)
n_cols = 20
process_data(file_name, n_cols)
n_cols = 30
process_data(file_name, n_cols)
n_cols = 40
process_data(file_name, n_cols)
n_cols = 50
process_data(file_name, n_cols)
n_cols = 60
process_data(file_name, n_cols)



winsound.Beep(3000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒
winsound.Beep(2000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒

winsound.Beep(4000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒
winsound.Beep(3000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒
winsound.Beep(1000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒
