using System;
using System.Collections.Generic;
using System.Text;

namespace MachineLearning
{
    public class Matrix
    {
        readonly float[,] Data;

        #region Properties

        public int Width { get; private set; }
        public int Height { get; private set; }

        #endregion

        #region Constructors

        public Matrix(int width, int height)
        {
            Data = new float[width, height];

            Width = width;
            Height = height;
        }

        #endregion

        #region Indexers

        public float this[int x, int y]
        {
            get { return Data[x, y]; }
            set { Data[x, y] = value; }
        }

        #endregion

        #region Math

        static Matrix LoopThrough(Matrix v1, Matrix v2, Func<float, float, float> action)
        {
            if (v1.Width != v2.Width || v1.Height != v2.Height)
                throw new Exception("Size of matrices are not equal");

            var result = new Matrix(v1.Width, v1.Height);

            for (int x = 0; x < v1.Width; x++)
            {
                for (int y = 0; y < v1.Height; y++)
                {
                    result[x, y] = action(v1[x, y], v2[x, y]);
                }
            }

            return result;
        }

        public static Matrix operator + (Matrix v1, Matrix v2) => LoopThrough(v1, v2, (x, y) => x + y);
        public static Matrix operator - (Matrix v1, Matrix v2) => LoopThrough(v1, v2, (x, y) => x - y);
        public static Matrix operator * (Matrix v1, float value) => LoopThrough(v1, v1, (x, y) => x * value);
        public static Matrix operator / (Matrix v1, float value) => LoopThrough(v1, v1, (x, y) => x / value);

        #endregion

        #region Linq

        public float[,] ToArray()
        {
            var result = new float[Width, Height];

            for (int x = 0; x < Width; x++)
            {
                for (int y = 0; y < Height; y++)
                {
                    result[x, y] = this[x, y];
                }
            }

            return result;
        }

        public override string ToString()
        {
            var sb = new StringBuilder();

            for (int y = 0; y < Height; y++)
            {
                sb.Append("[");

                for (int x = 0; x < Width - 1; x++)
                {
                    sb.Append(Data[x, y] + ", ");
                }

                sb.Append(Data[Width - 1, y] + "]");
                sb.AppendLine();
            }

            return sb.ToString();
        }

        #endregion
    }
}
