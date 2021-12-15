using System;

namespace MachineLearning
{
    class Program
    {
	// Neural network example for predicting the output of an and gate
        static void Main()
        {
            var nn = new NeuralNetwork(2, 1);

            var inputs = new Vector[]
            {
                new Vector(0f, 0f),
                new Vector(0f, 1f),
                new Vector(1f, 0f),
                new Vector(1f, 1f)
            };
            var outputs = new Vector[]
            {
                new Vector(0f),
                new Vector(0f),
                new Vector(0f),
                new Vector(1f)
            };

            Console.WriteLine("Training network 10,000 times..");
            nn.Train(inputs, outputs, 10000, 0.4f, 4);
            Console.WriteLine("Training complete");
            Console.WriteLine("Cost of network: " + nn.Cost(inputs, outputs));

            Console.WriteLine();

            var input = new Vector(1f, 1f);
            Console.WriteLine("Predicting: \n" + input);
            var prediction = nn.Predict(input);
            Console.WriteLine("Network output: \n" + prediction);
        }
    }
}
