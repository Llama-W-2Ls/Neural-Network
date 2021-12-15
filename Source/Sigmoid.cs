using System;
using System.Collections.Generic;
using System.Text;

namespace MachineLearning
{
    public static class Sigmoid
    {
        public static float Evaluate(float x)
        {
            return 1f / (1 + MathF.Exp(-x));
        }

        public static float EvaluateDerivative(float x)
        {
            return x * (1 - x);
        }
    }
}
