import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd


class KMeansClustering:

    def __init__(self, k=3, max_iter=100):
        self.k = k
        self.centroids = None
        self.number_of_iters = 0
        self.max_iter = max_iter
        self.labels = []

    @staticmethod
    def euclidian_distance(point, centroids):
        return np.sqrt(np.sum((centroids - point) ** 2, axis=1))

    def fit(self, X):

        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]

        for i in range(self.max_iter):
            # zliczanie iteracji
            self.number_of_iters += 1

            labels = []

            # obliczam odległości pomiędzy wszystkimi punktami a centroidami, wybieram najmniejsza odleglosc i przypisuje etykiete
            # taka jaki jest numer danego klastra

            for data_point in X:
                distances = KMeansClustering.euclidian_distance(data_point, self.centroids)
                clusters = np.argmin(distances)
                labels.append(clusters)

            # tworze array z etykiet
            labels = np.array(labels)

            # indeksy punktów które przynależą do danego klastra [[]]
            cluster_indices = []

            # przypisywanie indeksów punktów do klastrów
            for i in range(self.k):
                cluster_indices.append(np.argwhere(labels == i))

            cluster_centers = []
            wcss = 0

            # obliczamy wcss, oraz średnią ze wszystkich punktów (dla każdej z cech) aby ustalić nowe położenie centroidu
            for i, indices in enumerate(cluster_indices):
                cluster_centers.append(np.mean(X[indices], axis=0))
                wcss += np.sum(np.power(self.euclidian_distance(X[indices], np.array(cluster_centers[i])), 2))

            # sprawdzamy warunek stopu, jeżeli etykiety się nie zmieniły kończymy
            if np.array_equal(np.array(self.labels), labels):
                break
            else:
                self.centroids = np.concatenate(cluster_centers)
                self.labels = labels



        return self.labels, self.number_of_iters, wcss

    # def calculate_wcss(self, X):
    #     wcss = 0
    #     for k in range(self.k):
    #         cluster_points = X[self.labels == k]
    #         centroid = self.centroids[k]
    #         wcss += np.sum(np.linalg.norm(cluster_points - centroid, axis=1)**2)
    #     return wcss


class KnnClassifier:
    def __init__(self, k=3):
        self.k = k
        self.x_train = None
        self.y_train = None

    @staticmethod
    def euclidian_distance(new_point, points):
        return np.sqrt(np.sum((points - new_point) ** 2))

    def fit(self, x_train, y_train):
        self.x_train = self.normalize_data(np.array(x_train))
        self.y_train = np.array(y_train)

    def predict(self, X):
        X = np.array(X)
        X = self.normalize_data(X)
        labels = [self._predict(x) for x in X]
        labels = np.array(labels)
        return np.concatenate(labels[:][:])
    # TODO bez sortowania
    def _predict(self, x):
        # obliczamy dystans od każdego punktu
        labels = []
        labeled_distances = [(self.euclidian_distance(x, X_train), y_train) for X_train, y_train in
                             zip(self.x_train, self.y_train)]
        # sortujemy po dystansie - rosnąco
        labeled_distances.sort(key=lambda x: x[0])

        labels.append(self._get_neighbors(labeled_distances))

        return labels

    def _get_neighbors(self, distances):
        neighbors = distances[0:self.k]
        neighbors = np.array(neighbors)
        label_info = {}
        # tworzymy słownik label: {ilosc_wystapien, suma_dystansu}
        for neighbor in neighbors:
            label = neighbor[1]
            distance = neighbor[0]
            if label in label_info:
                label_info[label] = (label_info[label][0] + 1, label_info[label][1] + distance)
            else:
                label_info[label] = (1, distance)
        # sortujemy słownik ilość sąsiadów - malejąco, dystans - rosnąco
        sorted_labels = sorted(label_info.items(), key=lambda x: (-x[1][0], x[1][1]))
        return sorted_labels[0][0]

    @staticmethod
    def accuracy(y_true, y_pred):
        correct = np.sum(y_true == y_pred)
        total = len(y_true)
        return correct / total

    @staticmethod
    def normalize_data(data):
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))


def knn_helper(data_train, data_train_labels, data_test, data_test_labels, best_k):
    knn = KnnClassifier(k=best_k)
    knn.fit(data_train, data_train_labels)
    labels = knn.predict(data_test)
    print('\nSumaryczny wynik klasyfikacji dla najlepszego k: \n', knn.accuracy(data_test_labels, labels)*100, '%')

    cm = confusion_matrix(data_test_labels, labels[:][:])

    df_cm = pd.DataFrame(cm, index=[i for i in ['setosa', 'versicolor', 'virginica']],
                         columns=[i for i in ['setosa', 'versicolor', 'virginica']])
    print('\nMacierz pomyłek: ')
    print(df_cm)

    accuracy_list = []
    for i in range(1, 16):
        knn = KnnClassifier(k=i)
        knn.fit(data_train, data_train_labels)
        labels = knn.predict(data_test)
        accuracy_list.append(knn.accuracy(data_test_labels, labels))

    accuracy_list = np.array(accuracy_list)
    accuracy_list = accuracy_list * 100
    print('\nLista sumarycznych wyników klasyfikacji: ')
    print(accuracy_list)

    plt.plot(np.arange(1, len(accuracy_list) + 1, 1), accuracy_list)
    plt.xlabel("Wartość k (liczba najbliższych sąsiadów)")
    plt.ylabel("Sumaryczny wynik klasyfikacji [%]")
    plt.show()


data_test = np.loadtxt('data_test.csv', delimiter=',')
data_train = np.loadtxt('data_train.csv', delimiter=',')
data = np.loadtxt('data.csv', delimiter=',')

# wykres WCSS
wcss_list = []
iters_list = []

for i in range(2, 11):
    wcss_kmeans = KMeansClustering(k=i)
    labels, numbers_of_iters, wcss = wcss_kmeans.fit(data[:, 0:4])
    iters_list.append(numbers_of_iters)
    wcss_list.append(wcss)

wccs_number_of_clusters = np.arange(start=1, stop=11, step=1)
plt.plot(range(2, 11), wcss_list, marker='o')
plt.xlabel('Liczba klastrów')
plt.ylabel('WCSS')
plt.show()
print('\nLiczba iteracji dla k od 2 do 10: \n', str(iters_list))

my_kmeans = KMeansClustering(k=3)
labels, numbers_of_iters, wcss = my_kmeans.fit(data[:, 0:4])
print('\nWCSS: \n', wcss)
print('\nLiczba iteracji: \n', numbers_of_iters)

# wykresy K-means

figure, axis = plt.subplots(3, 2)
figure.set_size_inches(15, 18)

axis[0, 0].set_title('Długość działki kielicha do Szerokość działki kielicha')
axis[0, 0].set_xlabel('Długość działki kielicha [cm]')
axis[0, 0].set_ylabel('Szerokość działki kielicha [cm]')
axis[0, 0].scatter(data[:, 0], data[:, 1], c=labels, cmap='rainbow')
axis[0, 0].scatter(my_kmeans.centroids[:, 0], my_kmeans.centroids[:, 1], color='black', marker='*', s=150)

axis[0, 1].set_title('Długość działki kielicha do Długość płatka')
axis[0, 1].set_xlabel('Długość działki kielicha [cm]')
axis[0, 1].set_ylabel('Długość płatka [cm]')
axis[0, 1].scatter(data[:, 0], data[:, 2], c=labels, cmap='rainbow')
axis[0, 1].scatter(my_kmeans.centroids[:, 0], my_kmeans.centroids[:, 2], color='black', marker='*', s=150)

axis[1, 0].set_title('Długość działki kielicha do Szerokość płatka')
axis[1, 0].set_xlabel('Długość działki kielicha [cm]')
axis[1, 0].set_ylabel('Szerokość płatka [cm]')
axis[1, 0].scatter(data[:, 0], data[:, 3], c=labels, cmap='rainbow')
axis[1, 0].scatter(my_kmeans.centroids[:, 0], my_kmeans.centroids[:, 3], color='black', marker='*', s=150)

axis[1, 1].set_title('Szerokość działki kielicha do Długość płatka')
axis[1, 1].set_xlabel('Szerokość działki kielicha [cm]')
axis[1, 1].set_ylabel('Długość płatka [cm]')
axis[1, 1].scatter(data[:, 1], data[:, 2], c=labels, cmap='rainbow')
axis[1, 1].scatter(my_kmeans.centroids[:, 1], my_kmeans.centroids[:, 2], color='black', marker='*', s=150)

axis[2, 0].set_title('Szerokość działki kielicha do Szerokość płatka')
axis[2, 0].set_xlabel('Szerokość działki kielicha [cm]')
axis[2, 0].set_ylabel('Szerokość płatka [cm]')
axis[2, 0].scatter(data[:, 1], data[:, 3], c=labels, cmap='rainbow')
axis[2, 0].scatter(my_kmeans.centroids[:, 1], my_kmeans.centroids[:, 3], color='black',marker='*', s=150)

axis[2, 1].set_title('Długość płatka do Szerokość płatka')
axis[2, 1].set_xlabel('Długość płatka [cm]')
axis[2, 1].set_ylabel('Szerokość płatka [cm]')
axis[2, 1].scatter(data[:, 2], data[:, 3], c=labels, cmap='rainbow')
axis[2, 1].scatter(my_kmeans.centroids[:, 2], my_kmeans.centroids[:, 3], color='black', marker='*', s=150)

plt.show()

# Klasyfikacja - wszystkie cztery cechy
knn_helper(data_train[:, 0:4], data_train[:, 4], data_test[:, 0:4], data_test[:, 4], 6)

# Klasyfikacja – długość działki kielicha do szerokości działki kielicha
print('Klasyfikacja – długość działki kielicha do szerokości działki kielicha')
knn_helper(data_train[:, [0, 1]], data_train[:, 4], data_test[:, [0, 1]], data_test[:, 4], 15)

# Klasyfikacja – długość działki kielicha do długości płatka
print('Klasyfikacja – długość działki kielicha do długości płatka')
knn_helper(data_train[:, [0, 2]], data_train[:, 4], data_test[:, [0, 2]], data_test[:, 4], 3)

# Klasyfikacja – długość działki kielicha do szerokości płatka
print('Klasyfikacja – długość działki kielicha do szerokości płatka')
knn_helper(data_train[:, [0, 3]], data_train[:, 4], data_test[:, [0, 3]], data_test[:, 4], 5)

# Klasyfikacja – szerokość działki kielicha do długości płatka
print('Klasyfikacja – szerokość działki kielicha do długości płatka')
knn_helper(data_train[:, [1, 2]], data_train[:, 4], data_test[:, [1, 2]], data_test[:, 4], 5)

# Klasyfikacja – szerokość działki kielicha do szerokości płatka
print('Klasyfikacja – szerokość działki kielicha do szerokości płatka')
knn_helper(data_train[:, [1, 3]], data_train[:, 4], data_test[:, [1, 3]], data_test[:, 4], 3)

# Klasyfikacja – długość płatka do szerokości płatka
print('Klasyfikacja – długość płatka do szerokości płatka')
knn_helper(data_train[:, [2, 3]], data_train[:, 4], data_test[:, [2, 3]], data_test[:, 4], 2)
