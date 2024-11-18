import cv2
import numpy as np
import matplotlib.pyplot as plt

POPULATION_SIZE = 160  # 集団のサイズ
SUVIVAL_RATE = 0.5  # 生存率

def initialize_population(base_image, population_size=10):
    """
    拡大画像の初期集団を生成

    Parameters:
    - base_image: 拡大画像の基準となる画像（2次元配列）

    Returns:
    - initial_population: 初期集団（リスト形式）
    """
    height, width = base_image.shape
    large_image_shape = (height * 2, width * 2)
    initial_population = []

    # 最近隣内挿法に基づく拡大画像を基準として生成
    base_expanded = np.repeat(np.repeat(base_image, 2, axis=0), 2, axis=1)

    for _ in range(population_size):
        candidate_deltas = np.zeros(large_image_shape, dtype=np.float32)
        # 2x2のブロックごとに変位を加える
        for i in range(0, large_image_shape[0], 2):
            for j in range(0, large_image_shape[1], 2):
                neighborhood = base_expanded[max(0, i-1):min(large_image_shape[0], i+3),
                                             max(0, j-1):min(large_image_shape[1], j+3)]
                G_min, G_max = neighborhood.min(), neighborhood.max()
                delta_range_min = G_min - base_expanded[i:i+2, j:j+2]
                delta_range_max = G_max - base_expanded[i:i+2, j:j+2]
                delta = np.random.uniform(delta_range_min, delta_range_max)
                delta = delta - delta.mean() # 変位の合計をゼロに調整
                candidate_deltas[i:i+2, j:j+2] = delta
        initial_population.append(candidate_deltas)

    return initial_population, base_expanded

def calculate_edge_strength(image):
    """
    画像の1次微分によるエッジ強度 E(k) を計算する。
    """
    image = image.astype(np.float32)
    # エッジ強度の計算 (Prewittフィルタを使用)
    delta_h = (image[2:, 1:-1] + image[2:, :-2] + image[2:, 2:] -
               image[:-2, 1:-1] - image[:-2, :-2] - image[:-2, 2:])
    delta_v = (image[1:-1, 2:] + image[:-2, 2:] + image[2:, 2:] -
               image[1:-1, :-2] - image[:-2, :-2] - image[2:, :-2])

    E_k = np.sum(np.sqrt(delta_h**2 + delta_v**2))
    return E_k

def calculate_noise(image):
    """
    画像の2次微分によるノイズ成分 L(k) を計算する。
    """
    # 2次微分 (Laplacianオペレータを使用)
    l_x = (image[:-2, :-2] + image[:-2, 1:-1] + image[:-2, 2:] +
           image[1:-1, :-2] + image[1:-1, 2:] +
           image[2:, :-2] + image[2:, 1:-1] + image[2:, 2:] - 8 * image[1:-1, 1:-1])

    L_k = np.sum(np.sqrt(l_x**2))
    return L_k

def calculate_fitness(delta_image, base_expanded=None):
    # デルタ値を用いて実際の画像を生成
    if base_expanded is not None:
        image = generate_image_from_deltas(delta_image, base_expanded)
    else:
        image = delta_image
    # 以下、エッジ強度とノイズ成分を計算
    E_k = calculate_edge_strength(image)
    L_k = calculate_noise(image)
    F_k = E_k / (L_k + 1e-5)
    return F_k

def selection(population, fitness_scores, survival_rate=SUVIVAL_RATE):
    """
    適応度に基づき、上位S%の個体を選択して次世代を生成する。
    """
    # 個体数と生存する個体数の計算
    population_size = len(population)
    num_survivors = int(population_size * survival_rate)

    # 適応度に基づいてランク付けし、上位S%の個体を選択
    sorted_indices = np.argsort(fitness_scores)[::-1]
    survivors = [population[i] for i in sorted_indices[:num_survivors]]

    # 淘汰された個体数分の新しい個体を生成
    new_population = survivors.copy()
    num_offsprings = population_size - num_survivors

    # `survivors` 内からランダムにインデックスを選択して親を選択
    for _ in range(num_offsprings):
        parent_indices = np.random.choice(len(survivors), size=2, replace=False)
        parent1, parent2 = survivors[parent_indices[0]], survivors[parent_indices[1]]
        child = crossover(parent1, parent2)
        new_population.append(child)

    return new_population

def crossover(parent1, parent2):
    """
    2×2のブロック単位で交叉を行い、子供の染色体（デルタ値）を生成する。
    """
    child = np.zeros_like(parent1)
    height, width = parent1.shape

    for i in range(0, height, 2):
        for j in range(0, width, 2):
            parent_block_A = parent1[i:i+2, j:j+2].flatten()
            parent_block_B = parent2[i:i+2, j:j+2].flatten()

            # 各ブロックの近傍遺伝子（デルタ値）の最小値 G_min と最大値 G_max を取得
            neighborhood = np.concatenate([
                parent1[max(0, i-1):min(height, i+3), max(0, j-1):min(width, j+3)].flatten(),
                parent2[max(0, i-1):min(height, i+3), max(0, j-1):min(width, j+3)].flatten()
            ])
            G_min, G_max = neighborhood.min(), neighborhood.max()

            success = False
            attempts = 0
            while not success and attempts < 100:
                selected_indices = np.random.choice(4, 3, replace=False)
                e1 = parent_block_A[selected_indices[0]]
                e2 = parent_block_B[selected_indices[1]]
                e3 = parent_block_A[selected_indices[2]]
                e4 = -(e1 + e2 + e3)

                if G_min <= e4 <= G_max:
                    success = True
                else:
                    attempts += 1

            if not success:
                # デルタ値の範囲外の場合、クリップして総和がゼロになるように調整
                e4 = np.clip(e4, G_min, G_max)

            child_block = np.array([e1, e2, e3, e4])
            np.random.shuffle(child_block)
            child[i:i+2, j:j+2] = child_block.reshape(2, 2)

    return child


def genetic_algorithm(initial_population, base_expanded, generations=100, survival_rate=0.5):
    """
    GAを実行し、進化完了の判定を行う。

    Parameters:
    - initial_population: 初期集団（リスト形式）
    - generations: 最大世代数（デフォルトは100）
    - survival_rate: 生存率S%（0〜1の間の小数、デフォルトは0.5）

    Returns:
    - best_individual: 最終解となる最良の個体
    - best_fitness: 最良の個体の適応度
    """
    population = initial_population
    best_fitness_history = []
    max_fitness_stable_count = 0  # 進化完了の判定に使用
    tolerance = 1e-5  # 許容誤差を設定

    for generation in range(generations):
        # 各個体の適応度を計算
        fitness_scores = [calculate_fitness(ind, base_expanded) for ind in population]

        # 集団内の最良の適応度とその個体を取得
        max_fitness = max(fitness_scores)
        best_individual = population[np.argmax(fitness_scores)]

        # 最良の適応度を履歴に追加
        best_fitness_history.append(max_fitness)

        # 進化完了の判定
        if generation > 0:
            if abs(max_fitness - best_fitness_history[-2]) < tolerance:
                max_fitness_stable_count += 1
            else:
                max_fitness_stable_count = 0

            if max_fitness_stable_count >= 10:
                print(f"進化が完了しました。世代数: {generation + 1}, 最大適応度: {max_fitness:.6f}")
                break
        else:
            max_fitness_stable_count = 0  # 初回はカウンタをリセット

        # 次世代の集団を生成
        population = selection(population, fitness_scores, survival_rate=survival_rate)

        if generation % 10 == 0:
            print(f"第{generation + 1}世代: 最大適応度: {max_fitness:.6f}")

    return best_individual, max_fitness

def generate_image_from_deltas(delta_image, base_expanded):
    image = base_expanded + delta_image
    return image

if __name__ == '__main__':
    # 画像の読み込み（グレースケール）
    original_image = cv2.imread('imgs/sample2.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    original_image = cv2.resize(original_image, (128, 128)) # 画像サイズをリサイズ

    if original_image is None:
        raise FileNotFoundError("指定された画像ファイルが見つかりません")

    # 縮小画像を生成
    scale = 0.5  # 縮小倍率
    small_image = cv2.resize(original_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # 拡大画像のサイズを設定
    height, width = small_image.shape
    large_image_shape = (height * 2, width * 2)

    # 最近隣内挿法で生成した拡大画像
    nearest_neighbor_image = cv2.resize(small_image, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 3次畳み込み内挿法で生成した拡大画像
    bicubic_image = cv2.resize(small_image, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_CUBIC)


    ###################### ここからGAの実行 ######################
    print("Start GA...")
    # 集団の初期化
    population_size = POPULATION_SIZE
    initial_population, base_expanded = initialize_population(small_image, population_size)

    # GAの実行
    best_individual, best_fitness = genetic_algorithm(initial_population, base_expanded, generations=1000, survival_rate=0.5)

    best_individual = generate_image_from_deltas(best_individual, base_expanded)

    # 結果の表示
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    axes[0].imshow(original_image, cmap='gray')
    original_image_fitness = calculate_fitness(original_image)
    axes[0].set_title(f'Original Image (Fitness: {original_image_fitness:.2f})')

    axes[1].imshow(nearest_neighbor_image, cmap='gray')
    nearest_neighbor_image_fitness = calculate_fitness(nearest_neighbor_image)
    axes[1].set_title(f'Nearest Neighbor (Fitness: {nearest_neighbor_image_fitness:.2f})')


    axes[2].imshow(bicubic_image, cmap='gray')
    bicubic_image_fitness = calculate_fitness(bicubic_image)
    axes[2].set_title(f'Bicubic Interpolation (Fitness: {bicubic_image_fitness:.2f})')

    axes[3].imshow(best_individual, cmap='gray')
    axes[3].set_title(f'Enlarged Image (Fitness: {best_fitness:.2f})')

    for ax in axes:
        ax.axis('off')  # 軸を非表示にする

    plt.tight_layout()
    plt.show()

    cv2.imwrite('imgs/enlarged_image.jpg', best_individual)

