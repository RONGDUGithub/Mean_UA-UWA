import winsound
from mean_multiple_services import *


def run_experiment(file_name, mechanisms, epsilons):
    data_array = pd.read_excel(file_name)
    data0 = data_array.to_numpy()
    data0 = data0.flatten()
    max_val = np.max(data0)
    min_val = np.min(data0)
    data0 = 2 * (data0 - min_val) / (max_val - min_val) - 1
    mean_ground_truth = np.mean(data0)
    user_number = len(data0)

    min_data0 = np.min(data0)
    max_data0 = np.max(data0)

    # 重复运行的次数
    num_runs = 2  # 10
    result_run = np.zeros(len(mechanisms)+2)

    for run in range(num_runs):

        d_default = 40
        results, nor_noise, mean_list = multi_mechanism_process(mechanisms, data0, min_data0, max_data0, epsilons)

        midpoints_minus1_to_1 = torch.tensor(np.linspace(-1, 1 - 2 / d_default, d_default) + 1 / d_default).cuda()
        batch_size = user_number // 100
        final_weight_sum = 0

        for i in tqdm(range(1, int(user_number / batch_size + 1))):  # user_number + 1
            batch_noise = torch.vstack(results)[:, (i - 1) * batch_size: i * batch_size]
            batch_nor = torch.vstack(nor_noise)[:, (i - 1) * batch_size: i * batch_size]
            mean_tensor = compute_weighted_mean(
                mechanisms,
                epsilons,
                batch_size,
                batch_noise,
                batch_nor,
                d_default,
                midpoints_minus1_to_1
            )
            final_weight_sum += mean_tensor

        mean_tensor = final_weight_sum / (user_number)
        mean_aggre = mean_tensor.detach().cpu().numpy()  # .to('cpu')
        mean_baseline = np.mean(mean_list)
        mean_list.append(mean_baseline)
        mean_list.append(mean_aggre)
        mean_list2 = np.array(mean_list, dtype=float)
        MSE_aggre = ((mean_list2 - mean_ground_truth) ** 2)
        print('MSE_aggre', MSE_aggre)
        result_run += MSE_aggre

    MSE_final = result_run/num_runs
    return MSE_final

file_name = 'retirement.xlsx'
mechanisms = ['duchi', 'laplace', 'piecewise', 'sw']
epsilons = [0.1, 0.1, 0.1, 0.1]
MSE_final = run_experiment(file_name, mechanisms, epsilons)
base_name = os.path.splitext(file_name)[0]
output_filename = './group2/Group2_{}_A.txt'.format(base_name)
with open(output_filename, 'w') as f:
    f.write('Mechanisms:\n')
    f.write(str(mechanisms) + '\n')
    f.write('Epsilons:\n')
    f.write(str(epsilons) + '\n')  # 直接转换list为字符串
    f.write('MSE:\n')
    f.write(str(MSE_final.tolist()) + '\n')  # 假设MSE_final是numpy数组

winsound.Beep(3000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒
winsound.Beep(2000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒

winsound.Beep(4000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒
winsound.Beep(3000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒
winsound.Beep(1000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒
