using System;
using System.Collections.Generic;
using System.Linq;

namespace MachineLearning
{
    public class NeuralNetwork
    {
        Matrix[] Weights;
        Vector[] Biases;

        #region Properties

        public int[] LayerSizes { get; private set; }

        #endregion

        #region Constructors

        /// <summary>
        /// Initializes the weights and biases of the neural network
        /// with regards to the set sizes per layer
        /// </summary>
        /// <param name="layerSizes"></param>
        public NeuralNetwork(params int[] layerSizes)
        {
            LayerSizes = layerSizes;
            InitializeLayers(new Random());
        }

        /// <summary>
        /// Copies the weights and biases from another
        /// neural network into itself
        /// </summary>
        /// <param name="nn">The network to copy</param>
        public NeuralNetwork(NeuralNetwork nn)
        {
            LayerSizes = nn.LayerSizes;
            Weights = nn.Weights;
            Biases = nn.Biases;
        }

        #endregion

        #region Initialization

        /// <summary>
        /// Initialize weight and bias matrices/vectors
        /// </summary>
        void InitializeLayers(Random rand)
        {
            Weights = new Matrix[LayerSizes.Length - 1];
            Biases = new Vector[LayerSizes.Length - 1];

            for (int i = 0; i < LayerSizes.Length - 1; i++)
            {
                Weights[i] = new Matrix(LayerSizes[i + 1], LayerSizes[i]);
                Biases[i] = new Vector(LayerSizes[i + 1]);

                // Initialize random values for weight matrices
                for (int x = 0; x < Weights[i].Width; x++)
                {
                    for (int y = 0; y < Weights[i].Height; y++)
                    {
                        Weights[i][x, y] = GetRandomValue(rand, LayerSizes[i + 1]);
                    }
                }

                // Initialize random values for bias vectors
                for (int j = 0; j < Biases[i].Length; j++)
                {
                    Biases[i][j] = GetRandomValue(rand, LayerSizes[i + 1]);
                }
            }
        }

        /// <summary>
        /// Calculates random value in the standard normal distribution (-3 to 3)
        /// and divides by sqrt(layerSize) for a good starting weight/bias.
        /// </summary>
        float GetRandomValue(Random rand, int layerSize)
        {
            var num = rand.Next(-2, 3) + (rand.NextDouble() * 2 - 1);
            return (float)num / MathF.Pow(layerSize, 0.5f);
        }

        #endregion

        #region Evaluation

        /// <summary>
        /// Feeds an input through each layer and calculates
        /// the output of the neurons in each layer
        /// </summary>
        Vector[] FeedForward(Vector input)
        {
            var results = new Vector[LayerSizes.Length];
            results[0] = input;

            for (int i = 1; i < results.Length; i++)
            {
                results[i] = new Vector(LayerSizes[i]);

                for (int j = 0; j < LayerSizes[i]; j++)
                {
                    float value = 0f;
                    for (int k = 0; k < LayerSizes[i - 1]; k++)
                    {
                        value += Weights[i - 1][j, k] * results[i - 1][k];
                    }

                    results[i][j] = Sigmoid.Evaluate(value + Biases[i - 1][j]);
                }
            }

            return results;
        }

        /// <summary>
        /// Feed an input through the network
        /// </summary>
        /// <param name="input"></param>
        /// <returns>The final output</returns>
        public Vector Predict(Vector input)
        {
            var predictions = FeedForward(input);
            return predictions[^1];
        }

        /// <summary>
        /// Calculates the mean-squared-error (MSE) of the network
        /// (sum of (all outputs - all expected outputs)^2) / total size)
        /// </summary>
        public float Cost(Vector[] inputs, Vector[] expectedOutputs)
        {
            var totalCost = 0f;

            for (int i = 0; i < inputs.Length; i++)
            {
                var output = Predict(inputs[i]);

                for (int j = 0; j < output.Length; j++)
                {
                    totalCost += MathF.Pow(output[j] - expectedOutputs[i][j], 2);
                }
            }

            return totalCost / inputs.Length;
        }

        #endregion

        #region Training

        /// <summary>
        /// Trains the neural network with the mini-batch stochastic gradient descent algorithm
        /// and backpropagation
        /// </summary>
        /// <param name="inputs">Training data input</param>
        /// <param name="expectedOutputs">Training data output</param>
        /// <param name="iterations">Number of times to train same set of data</param>
        /// <param name="learningRate">How small/large should adjustments to learning be</param>
        /// <param name="batchSize">The size of a mini-batch</param>
        public void Train(Vector[] inputs, Vector[] expectedOutputs, int iterations, float learningRate, int batchSize)
        {
            #region Package training data together

            var trainingData = new Tuple<Vector, Vector>[inputs.Length];

            for (int i = 0; i < inputs.Length; i++)
            {
                trainingData[i] = new Tuple<Vector, Vector>(inputs[i], expectedOutputs[i]);
            }

            #endregion

            #region Train iteratively

            var rand = new Random();

            for (int i = 0; i < iterations; i++)
            {
                var miniBatch = TakeRandom(trainingData, rand, batchSize);
                TrainBatch(miniBatch, learningRate);
            }

            #endregion
        }

        /// <summary>
        /// Trains a batch of data by finding the average weight and bias
        /// deltas for each sample in the batch, and adjusting the network's
        /// weights and biases by those values
        /// </summary>
        void TrainBatch(IEnumerable<Tuple<Vector, Vector>> batch, float learningRate)
        {
            #region Initialize weight/bias delta matrices/vectors

            var averageWeightDeltas = new Matrix[Weights.Length];
            var averageBiasDeltas = new Vector[Biases.Length];

            for (int i = 0; i < Weights.Length; i++)
            {
                averageWeightDeltas[i] = new Matrix(Weights[i].Width, Weights[i].Height);
                averageBiasDeltas[i] = new Vector(Biases[i].Length);
            }

            #endregion

            #region Calculate average weight/bias adjustments

            foreach (var sample in batch)
            {
                BackPropagate(sample.Item1, sample.Item2, out Matrix[] weightDeltas, out Vector[] biasDeltas);

                for (int i = 0; i < weightDeltas.Length; i++)
                {
                    averageWeightDeltas[i] += weightDeltas[i];
                    averageBiasDeltas[i] += biasDeltas[i];
                }
            }

            #endregion

            #region Adjust weights and biases

            for (int i = 0; i < averageWeightDeltas.Length; i++)
            {
                Weights[i] += averageWeightDeltas[i] * learningRate;
                Biases[i] += averageBiasDeltas[i] * learningRate;
            }

            #endregion
        }

        /// <summary>
        /// Backpropagation algorithm to find the weight and bias deltas for
        /// a single sample in the network
        /// </summary>
        void BackPropagate(Vector input, Vector expectedOutput, out Matrix[] weightDeltas, out Vector[] biasDeltas)
        {
            #region Initialize weight and bias deltas

            weightDeltas = new Matrix[Weights.Length];
            biasDeltas = new Vector[Biases.Length];

            for (int i = 0; i < Weights.Length; i++)
            {
                weightDeltas[i] = new Matrix(Weights[i].Width, Weights[i].Height);
                biasDeltas[i] = new Vector(Biases[i].Length);
            }

            #endregion

            #region Back propagation

            // Implementation from: https://github.com/kipgparker/BackPropNetwork/blob/master/BackpropNeuralNetwork/Assets/NeuralNetwork.cs

            // Get values of neurons at all layers
            var outputs = FeedForward(input);

            #region Initialize gamma

            var gammaList = new List<float[]>();

            for (int i = 0; i < LayerSizes.Length; i++)
            {
                gammaList.Add(new float[LayerSizes[i]]);
            }

            float[][] gamma = gammaList.ToArray();

            #endregion

            #region Calculate gamma, weights and bias deltas on last layer of network

            for (int i = 0; i < outputs[^1].Length; i++)
                gamma[LayerSizes.Length - 1][i] = (outputs[^1][i] - expectedOutput[i]) * Sigmoid.EvaluateDerivative(outputs[^1][i]);

            // Calculates the 'w' and b' for the last layer in the network
            for (int i = 0; i < LayerSizes[^1]; i++)
            {
                biasDeltas[LayerSizes.Length - 2][i] -= gamma[LayerSizes.Length - 1][i];

                for (int j = 0; j < LayerSizes[^2]; j++)
                {
                    //learning
                    weightDeltas[LayerSizes.Length - 2][i, j] -= gamma[LayerSizes.Length - 1][i] * outputs[LayerSizes.Length - 2][j];
                }
            }

            #endregion

            #region Calculate gamma, weights and bias deltas on all hidden layers of network

            // Runs on all hidden layers
            for (int i = LayerSizes.Length - 2; i > 0; i--)
            {
                // Outputs
                for (int j = 0; j < LayerSizes[i]; j++)
                {
                    gamma[i][j] = 0;

                    for (int k = 0; k < gamma[i + 1].Length; k++)
                    {
                        gamma[i][j] += gamma[i + 1][k] * Weights[i][k, j];
                    }

                    //Calculate gamma
                    gamma[i][j] *= Sigmoid.EvaluateDerivative(outputs[i][j]);
                }

                // Iterate over outputs of layer
                for (int j = 0; j < LayerSizes[i]; j++)
                {
                    // Modify biases of network
                    biasDeltas[i - 1][j] -= gamma[i][j];

                    // Iterate over inputs to layer
                    for (int k = 0; k < LayerSizes[i - 1]; k++)
                    {
                        // Modify weights of network
                        weightDeltas[i - 1][j, k] -= gamma[i][j] * outputs[i - 1][k];
                    }
                }
            }

            #endregion

            #endregion
        }

        /// <summary>
        /// Takes a random sample from a collection of size 'takeCount'
        /// (used to initialize a random mini-batch before training)
        /// </summary>
        IEnumerable<T> TakeRandom<T>(IEnumerable<T> collection, Random rand, int takeCount)
        {
            var available = collection.Count();
            var needed = takeCount;

            foreach (var item in collection)
            {
                if (rand.Next(available) < needed)
                {
                    needed--;
                    yield return item;

                    if (needed == 0)
                    {
                        break;
                    }
                }

                available--;
            }
        }

        #endregion
    }
}
