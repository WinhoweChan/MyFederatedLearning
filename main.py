from server import run_exp
from federated_learning.my_dict import replace_1_with_7, replace_dog_with_cat
from federated_learning.my_dict import run_mnist, run_cifar10
from federated_learning.my_dict import fed_avg, my_function, simple_median, trimmed_mean, multi_krum, fools_gold

if __name__ == '__main__':
    NUM_WORKERS = 10
    FRAC_WORKERS = 1
    ATTACK_TYPE = "label_flipping"

    # REPLACE_METHOD = replace_1_with_7()
    # RULE = my_function()
    # DATASET = run_mnist()

    REPLACE_METHOD = replace_dog_with_cat()
    RULE = my_function()
    DATASET = run_cifar10()

    MALICIOUS_RATE = [0.3]  # 0, 0.1, 0.2, 0.3, 0.4, 0.5
    MALICIOUS_BEHAVIOR_RATE = 1

    for rate in MALICIOUS_RATE:
        run_exp(NUM_WORKERS, FRAC_WORKERS, ATTACK_TYPE, RULE, REPLACE_METHOD, DATASET, rate, MALICIOUS_BEHAVIOR_RATE)
