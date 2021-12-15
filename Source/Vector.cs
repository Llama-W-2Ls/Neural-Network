using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace MachineLearning
{
    public class Vector : IEnumerable
    {
        readonly float[] Data;

        #region Properties

        public int Length { get { return Data.Length; } }

        #endregion

        #region Constructors

        public Vector(int length)
        {
            Data = new float[length];
        }

        public Vector(params float[] values)
        {
            Data = values;
        }

        #endregion

        #region Indexers

        public float this[int index]
        {
            get { return Data[index]; }
            set { Data[index] = value; }
        }

        #endregion

        #region Math

        static Vector LoopThrough(Vector v1, Vector v2, Func<float, float, float> action)
        {
            if (v1.Length != v2.Length)
                throw new Exception("Size of vectors are not equal");

            var result = new Vector(v1.Length);

            for (int i = 0; i < v1.Length; i++)
            {
                result[i] = action(v1[i], v2[i]);
            }

            return result;
        }

        public static Vector operator + (Vector v1, Vector v2) => LoopThrough(v1, v2, (x, y) => x + y);
        public static Vector operator - (Vector v1, Vector v2) => LoopThrough(v1, v2, (x, y) => x - y);
        public static Vector operator * (Vector v1, float value) => LoopThrough(v1, v1, (x, y) => x * value);
        public static Vector operator / (Vector v1, float value) => LoopThrough(v1, v1, (x, y) => x / value);

        #endregion

        #region Linq

        public float[] ToArray()
        {
            var result = new float[Data.Length];

            for (int i = 0; i < Data.Length; i++)
            {
                result[i] = Data[i];
            }

            return result;
        }

        public override string ToString()
        {
            var sb = new StringBuilder();

            foreach (var item in Data)
            {
                sb.AppendLine("[" + item + "]");
            }

            return sb.ToString();
        }

        #endregion

        #region Enumerators

        public IEnumerator GetEnumerator()
        {
            return Data.GetEnumerator();
        }

        #endregion
    }
}
