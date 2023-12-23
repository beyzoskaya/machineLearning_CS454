import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import math

def load_data(file_path):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            data.append(np.array([float(row[0]), float(row[1])])) 
            labels.append(float(row[2]))
    return data, labels


def binary_cross_entropy(y_true, y_pred):
    bce = -(y_true * math.log(y_pred) + (1 - y_true) * math.log(1 - y_pred))
    return bce

def dot_product(v, w):
    v = np.array(v)
    w = np.array(w)

    # Case 1: Both v and w are vectors (1D arrays)
    if v.ndim == 1 and w.ndim == 1:
        if v.shape[0] != w.shape[0]:
            raise ValueError("Vectors must have the same length for dot product.")
        return custom_sum(v[i] * w[i] for i in range(len(v)))

    # Case 2: v is a 2D matrix and w is a vector
    elif v.ndim == 2 and w.ndim == 1:
        if v.shape[1] != w.shape[0]:
            raise ValueError("Number of columns in matrix must match vector length.")
        return np.array([custom_sum(v[i, j] * w[j] for j in range(w.shape[0])) for i in range(v.shape[0])])

    # Case 3: v is a vector and w is a 2D matrix
    elif v.ndim == 1 and w.ndim == 2:
        if len(v) != w.shape[1]:
            raise ValueError("Vector length must match the number of columns in the matrix.")
        return np.array([custom_sum(v[j] * w[i, j] for j in range(len(v))) for i in range(w.shape[0])])
    
    elif v.ndim == 1 and w.ndim == 2 and w.shape == 1:
        if len(v) != w.shape[1]:
            raise ValueError("Length of vector must match the number of columns in the matrix.")
        return custom_sum(v[j] * w[0, j] for j in range(len(v)))
        
    # Case 4: Both v and w are 2D matrices
    elif v.ndim == 2 and w.ndim == 2:
        if v.shape[1] != w.shape[0]:
            raise ValueError("The number of columns in the first matrix must match the number of rows in the second matrix.")
        return np.array([[custom_sum(v[i, k] * w[k, j] for k in range(v.shape[1])) for j in range(w.shape[1])] for i in range(v.shape[0])])

    else:
        raise ValueError("Unsupported array dimensions for dot product.")

def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def scalar_vector_product(scalar, vector):
    if not isinstance(scalar, (int, float)):
        raise ValueError("First argument must be a scalar.")
    return [scalar * v for v in vector]

def custom_sum(iterable):
    total = 0
    for item in iterable:
        total += item
    return total

def custom_exp(x):
    if isinstance(x, (float, int)):
        return math.exp(x)
    elif isinstance(x, np.ndarray):
        if x.ndim == 1:
            return np.array([math.exp(val) for val in x])
        elif x.ndim == 2:
            return np.array([[math.exp(val) for val in row] for row in x])
    else:
        raise ValueError("Unsupported type or array dimensions for exp function.")
    
class SingleLayerPerceptron:
    def __init__(self, input_size, learning_rate=0.01):
        self.weights = np.random.random_sample(input_size + 1) # burda sadece w var single oldugu icin
        print(f"Weights for single layer perceptron: {self.weights}")
        self.learning_rate = learning_rate
        #print(f"LR: {learning_rate}")

    
    def activation_sigmoid(self,x):
        return 1 / (1+ custom_exp(-x))
    
    def dot_product_single_layer(self, vector1, vector2):
        if len(vector1) != len(vector2):
            raise ValueError("Vectors must be of the same length.")
        return custom_sum(x * y for x, y in zip(vector1, vector2))
    
    def forward(self, inputs):
        #print(f"Inputs: {inputs}")
        weighted_sum = self.dot_product_single_layer(inputs, self.weights[:-1]) + self.weights[-1]
        return self.activation_sigmoid(weighted_sum)
    
    def train(self, training_inputs, labels, test_inputs, test_labels, epochs):
        loss_per_epoch = [] 
        self.train_losses = []
        self.test_losses = []

        for epoch in range(epochs):
            shuffle_data = list(zip(training_inputs, labels))
            random.shuffle(shuffle_data)
            total_loss = 0

            for inputs, y_true in zip(training_inputs, labels):
            #for inputs, y_true in shuffle_data:
                y_pred = self.forward(inputs)
                loss = binary_cross_entropy(y_true=y_true, y_pred=y_pred)
                total_loss += loss

                # Update weights
                #print(f"y_true: {y_true}")
                #print(f"y_pred: {y_pred}")

                # Update weights
                error = y_true - y_pred
                derivative = y_pred * (1 - y_pred)

                # Weight update for the input weights
                self.weights[:-1] += self.learning_rate * error * inputs * derivative

                # Weight update for the bias weight
                self.weights[-1] += self.learning_rate * error * derivative


            average_loss = total_loss / len(training_inputs)
            self.train_losses.append(average_loss)

            # Calculate test loss
            test_loss = 0
            for test_input, test_label in zip(test_inputs, test_labels):
                test_loss += binary_cross_entropy(y_pred=self.forward_test(test_input), y_true=test_label)
            test_loss /= len(test_inputs)
            self.test_losses.append(test_loss)

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {average_loss}, Test Loss: {test_loss}")
        self.plot_losses(self.train_losses, self.test_losses)
        return self.train_losses, self.test_losses

    
    def plot_loss(self, losses):
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Binary Cross Entropy Loss')
        plt.title(f'Single-Layer Perceptron')
        plt.savefig(f"./log_slp.png")
        plt.show()
    
    def plot_losses(self, train_losses, test_losses):
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Binary Cross Entropy Loss')
        plt.title(f'Single-Layer Perceptron')
        plt.legend()
        plt.savefig(f"plots/log_slp.png")
        plt.show()
    
    def forward_test(self, inputs):
        weighted_sum = self.dot_product_single_layer(inputs, self.weights[1:]) + self.weights[0]
        return self.activation_sigmoid(weighted_sum)

class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size=1, learning_rate=0.01):
        #self.weights_input_hidden = np.random.uniform(-0.01, 0.01, (input_size + 1, hidden_size))
        #print(f"Input hidden weight:: {self.weights_input_hidden}")
        #self.weights_hidden_output = np.random.uniform(-0.01, 0.01, (hidden_size + 1, output_size))
       # print(f"Hidden output weight: {self.weights_hidden_output}")
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        print(f"Hidden unit: {self.hidden_size}")
        self.w_hj, self.v_ih = self.initialize_weights(input_size, hidden_size,output_size)


    def initialize_weights(self, input_size,hidden_size, output_size):
        w_hj = np.random.uniform(-0.01, 0.01, (input_size + 1, hidden_size))
        #print(f"w_hj random initialization: {w_hj}")
        v_ih = np.random.uniform(-0.01, 0.01, (hidden_size + 1))
        #print(f"v_ih random initialization: {v_ih}")
        return w_hj, v_ih

    def sigmoid(self, x):
        return 1 / (1 + custom_exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self,inputs):
        #hidden_input = np.dot(inputs, self.w_hj[:-1])+ self.w_hj[-1]
        hidden_input = dot_product(transpose(self.w_hj[:-1]), inputs) + self.w_hj[-1]
        self.z_h = self.sigmoid(hidden_input)
        self.y_i = dot_product(self.z_h, self.v_ih[:-1]) + self.v_ih[-1]
        return self.sigmoid(self.y_i)
        
    def backward_pass(self, inputs, predicted_output, actual_output):
        
        error = actual_output - predicted_output
        delta_predicted_output = error * self.sigmoid_derivative(predicted_output)
        #error_hidden_layer = np.dot(delta_predicted_output, self.v_ih[:-1].T)
        error_hidden_layer = scalar_vector_product(delta_predicted_output, self.v_ih[:-1].T)
        #print(f"Delta predicted output shape: {delta_predicted_output.shape}")
        #print(f"v_ih T shape: {self.v_ih[:-1].T.shape} ")

        delta_hidden_layer = error_hidden_layer* self.sigmoid_derivative(self.z_h)
        #print(f"Predicted output: {predicted_output}")
        #print(f"Z_h: {self.z_h}")
        # print(f"Delta hidden layer: {delta_hidden_layer}")  

        #update weights
        #self.v_ih[:-1] += self.learning_rate *np.dot(self.z_h, delta_predicted_output) 
        scaled_gradient = scalar_vector_product(delta_predicted_output, self.z_h)
        self.v_ih[:-1] += self.learning_rate * np.array(scaled_gradient)

        #print(f"Shape for z_h: {self.z_h.shape}")
        #print(f"Shape for delta_predicted_output: {delta_predicted_output.shape} ")
        self.v_ih[-1] += self.learning_rate * delta_predicted_output
        self.w_hj[:-1] += self.learning_rate * dot_product(np.array([[x] for x in inputs]), [delta_hidden_layer])
        self.w_hj[-1] += self.learning_rate * delta_hidden_layer
    

    #def plot_losses(self, losses):
    #    plt.plot(losses)
    #    plt.savefig(f"./log_{self.hidden_size}.png")

    def forward_test(self, inputs):
        hidden_input = dot_product(transpose(self.w_hj[:-1]), inputs) + self.w_hj[-1]
        z_h = self.sigmoid(hidden_input)
        y_i = dot_product(z_h, self.v_ih[:-1]) + self.v_ih[-1]
        return self.sigmoid(y_i)
    
    def plot_loss(self, losses):
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Binary Cross Entropy Loss')
        plt.title(f'Multi-Layer Perceptron - Hidden Size: {self.hidden_size}')
        plt.savefig(f"./log_mlp_{self.hidden_size}.png")
        plt.show()

    def plot_losses(self, train_losses, test_losses):
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Binary Cross Entropy Loss')
        plt.title(f'Multi-Layer Perceptron - Hidden Size: {self.hidden_size}')
        plt.legend()
        plt.savefig(f"plots/log_mlp_{self.hidden_size}.png")
        plt.show()

    def train(self, training_inputs, labels, test_inputs, test_labels, epochs):
        self.train_losses = []
        self.test_losses = []

        for epoch in range(epochs):
            shuffle_data = list(zip(training_inputs, labels))
            random.shuffle(shuffle_data)
            total_loss = 0
            #for x, y in shuffle_data:
            for x, y in zip(training_inputs, labels):
                pred = self.forward(x)
                self.backward_pass(x, pred, y)
                loss = binary_cross_entropy(y_pred=pred, y_true=y)
                total_loss += loss
            average_loss = total_loss / len(training_inputs)
            self.train_losses.append(average_loss)

            # Calculate test loss
            test_loss = 0
            for test_input, test_label in zip(test_inputs, test_labels):
                test_loss += binary_cross_entropy(y_pred=self.forward_test(test_input), y_true=test_label)
            test_loss /= len(test_inputs)
            self.test_losses.append(test_loss)

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {average_loss}, Test Loss: {test_loss}")
            #train_losses.append(total_loss/len(training_inputs))
        #self.plot_losses(train_losses)
        self.plot_losses(self.train_losses, self.test_losses)
        return self.train_losses, self.test_losses
        

def plot_decision_boundary(model, data, labels, title, save_path=None):
    x_min, x_max = min(data[:, 0]) - 1, max(data[:, 0]) + 1
    y_min, y_max = min(data[:, 1]) - 1, max(data[:, 1]) + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    mesh_input = np.c_[xx.ravel(), yy.ravel()]
    predictions = []

    for point in mesh_input:
        prediction = model.forward(point)
        predictions.append(prediction)

    predictions = np.array(predictions).reshape(xx.shape)

    plt.figure(figsize=(10, 5))

    plt.plot(1, 2, 1)
    contour = plt.contourf(xx, yy, predictions, levels=20, alpha=0.8, cmap='coolwarm')
    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='coolwarm', edgecolors='k', marker='o', s=20, linewidth=1)
    plt.title('Decision Boundary')

    # Add a color bar for sigmoid values
    cbar = plt.colorbar(contour)
    cbar.set_label('Sigmoid Values')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.suptitle(title)
    plt.show()

def plot_overall_network_complexity_vs_error(hidden_sizes, slp_train_loss, slp_test_loss, mlp_train_losses, mlp_test_losses):
    plt.plot(hidden_sizes, [slp_train_loss[-1]] + mlp_train_losses, label='Train Loss')
    plt.plot(hidden_sizes, [slp_test_loss[-1]] + mlp_test_losses, label='Test Loss')
    plt.xlabel('Number of Hidden Units')
    plt.ylabel('Binary Cross Entropy Loss')
    plt.title('Overall Network Complexity vs Error')
    plt.legend()
    plt.savefig(f"plots/complexity.png")
    #plt.savefig(f"plots/complexity.png")
    plt.show()


def main():
    train_data, train_labels = load_data('./train.csv')
    test_data, test_labels = load_data('./test.csv')
    epochs = 1000
    hidden_layer_sizes = [2, 4, 8]
    hidden_layer_for_complexity = [0] + hidden_layer_sizes
    slp_train_losses, slp_test_losses = [], []
    mlp_train_losses, mlp_test_losses = [], []

    shuffle_plot_dir = "./shuffle_plots"
    os.makedirs(shuffle_plot_dir, exist_ok=True)

    plot_dir = "./plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Train and plot Multi-Layer Perceptrons
    for hidden_size in hidden_layer_sizes:
        print(f"\nTraining MLP with {hidden_size} hidden units")
        mlp = MultiLayerPerceptron(input_size=2, hidden_size=hidden_size, learning_rate=0.1)
        train_losses, test_losses = mlp.train(train_data, train_labels, test_data, test_labels, epochs)
        plot_decision_boundary(mlp, np.array(train_data), np.array(train_labels), title=f"MLP with {hidden_size} Hidden Units",
                               save_path=os.path.join(plot_dir, f"mlp_{hidden_size}_boundary.png"))
        mlp.plot_losses(train_losses, test_losses)
        mlp_train_losses.append(train_losses[-1])
        mlp_test_losses.append(test_losses[-1])

    # Train and plot Single-Layer Perceptron
    print("\nTraining Single-Layer Perceptron")
    slp = SingleLayerPerceptron(input_size=2, learning_rate=0.05)
    slp_train_losses, slp_test_losses = slp.train(train_data, train_labels, test_data, test_labels, epochs)
    plot_decision_boundary(slp, np.array(train_data), np.array(train_labels), title="Single-Layer Perceptron",
                           save_path=os.path.join(plot_dir, "slp_boundary.png"))
    slp.plot_losses(slp_train_losses, slp_test_losses)

    # Plot Overall Network Complexity vs Error
    plot_overall_network_complexity_vs_error(hidden_layer_for_complexity, slp_train_losses, slp_test_losses, mlp_train_losses,
                                             mlp_test_losses)

if __name__ == "__main__":
    main()


        