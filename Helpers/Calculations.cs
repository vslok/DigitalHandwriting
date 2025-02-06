using Microsoft.EntityFrameworkCore.Metadata.Conventions;
using Microsoft.Windows.Themes;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Markup;

namespace DigitalHandwriting.Helpers
{
    public static class Calculations
    {
        public static List<int> CalculateMedianValue(List<List<int>> values)
        {
            foreach (var list in values)
            {
                if (list.Count != values[0].Count)
                {
                    throw new Exception("Median value cannot be calculated");
                }
            }

            List<int> mediansList = new List<int>();

            for (int i = 0; i < values[0].Count; i++)
            {
                var valuesList = values.Select(list => list[i]).OrderBy(x => x).ToList();
                mediansList.Add(valuesList.ElementAt((int)Math.Ceiling(valuesList.Count / 2.0d)));
            }

            return mediansList;
        }

        public static double EuclideanDistance(List<double> vector1, List<double> vector2)
        {
            if (vector1.Count != vector2.Count)
                throw new ArgumentException("Vectors must be of same length.");

            double sumSquared = 0.0;
            for (int i = 0; i < vector1.Count; i++)
            {
                double diff = vector1[i] - vector2[i];
                sumSquared += diff * diff;
            }

            var result = Math.Sqrt(sumSquared);

            return Math.Round(result, 2);
        }

        public static double ManhattanDistance(List<double> vector1, List<double> vector2)
        {
            if (vector1.Count != vector2.Count)
                throw new ArgumentException("Vectors must be of same length.");

            double sumAbs = 0.0;
            for (int i = 0; i < vector1.Count; i++)
            {
                sumAbs += Math.Abs(vector1[i] - vector2[i]);
            }

            return Math.Round(sumAbs, 2);
        }

        public static double ManhattanDistance(double etPoint, double curPoint)
        { 
            return Math.Round(Math.Abs(etPoint - curPoint), 2);
        }

        public static double MahalanobisDistance(List<double> vector1, List<double> vector2, double[,] invCovMatrix)
        {
            if (vector1.Count != vector2.Count)
                throw new ArgumentException("Vectors must be of same length.");

            int n = vector1.Count;
            double[] delta = new double[n];
            for (int i = 0; i < n; i++)
            {
                delta[i] = vector1[i] - vector2[i];
            }

            // Perform delta^T * invCovMatrix * delta
            double[] temp = new double[n];
            for (int i = 0; i < n; i++)
            {
                temp[i] = 0.0;
                for (int j = 0; j < n; j++)
                {
                    temp[i] += invCovMatrix[i, j] * delta[j];
                }
            }

            double mahalanobis = 0.0;
            for (int i = 0; i < n; i++)
            {
                mahalanobis += delta[i] * temp[i];
            }

            return Math.Sqrt(mahalanobis);
        }

        public static double CosineSimilarity(List<int> etVector, List<int> curVector)
        {
            double dotProduct = 0;
            double normEnrollment = 0;
            double normCurrent = 0;

            for (int i = 0; i < etVector.Count; i++)
            {
                dotProduct += etVector[i] * curVector[i];
                normEnrollment += Math.Pow(etVector[i], 2);
                normCurrent += Math.Pow(curVector[i], 2);
            }

            return dotProduct / (Math.Sqrt(normEnrollment) * Math.Sqrt(normCurrent));
        }

        public static List<double> Normalize(List<int> vector)
        {
            var length = Math.Sqrt(vector.ConvertAll(el => Math.Pow(el, 2)).Sum());
            return vector.ConvertAll(el => el / length);
        }

        public static double ITAD(List<double> sample, List<List<double>> rawReferences)
        {
            var totalITAD = 0.0;
            for (var i = 0; i < rawReferences[0].Count; i++)
            {
                var keyValues = new List<double>();
                for (var j = 0; j < rawReferences.Count; j++)
                {
                    keyValues.Add(rawReferences[j][i]);
                }

                var sortedReference = keyValues.OrderBy(x => x).ToList();
                int count = sortedReference.Count(x => x <= sample[0]);
                double itad = (double)count / sortedReference.Count;
                totalITAD += itad;
            }

            return totalITAD / sample.Count;
        }

        public static List<List<T>> MatrixTransposing<T>(List<List<T>> matrix)
        {
            return matrix.SelectMany((l, i) => l.Select((d, j) => new { i, j, d }))
                         .GroupBy(l => l.j)
                         .Select(l => l.Select(ll => ll.d).ToList())
                         .ToList();
        }

        public static double Expectancy(List<int> intervals)
        {
            return intervals.Average();
        }

        public static double Dispersion(List<int> intervals, double expect)
        {
            return intervals.Select(interv => Math.Pow(interv - expect, 2)).Average();
        }
       
        public static double StandardDeviation(double disp)
        {
            return Math.Sqrt(disp);
        }
    }
}
